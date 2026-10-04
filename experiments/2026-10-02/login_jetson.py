"""User-operated login recovery. Passwords are forwarded, never stored or logged."""
import fcntl
import glob
import json
import os
from pathlib import Path
import re
import select
import sys
import subprocess
import termios
import tty
import urllib.request
from serial_session import clean_output, idle_shell_prompt

BASE = 'http://127.0.0.1:8769'

def saved_password():
    try:
        result = subprocess.run(['security','find-generic-password','-a','dimos','-s','a1z-jetson-console','-w'], capture_output=True, text=True, timeout=15)
        return result.stdout.rstrip('\n') if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def main():
    if not sys.stdin.isatty(): raise RuntimeError('请在 Terminal 中打开登录入口。')
    state = json.load(urllib.request.urlopen(BASE+'/api/status', timeout=3))
    if state['active']: raise RuntimeError('采集或准备进程正在运行，请回控制台；不会抢占连接。')
    lock = (Path(__file__).parent/'console.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ports = glob.glob('/dev/cu.usbserial-*')
    if len(ports)!=1: raise RuntimeError('未找到唯一的 Jetson 串口，请检查 USB-C。')
    fd = os.open(ports[0],os.O_RDWR|os.O_NOCTTY|os.O_NONBLOCK)
    old = termios.tcgetattr(fd); local = termios.tcgetattr(0); success = False
    try:
        fcntl.ioctl(fd, termios.TIOCEXCL)
        a = termios.tcgetattr(fd);a[0]=a[1]=a[3]=0
        a[2]=termios.CS8|termios.CREAD|termios.CLOCAL;a[4]=a[5]=termios.B115200
        termios.tcsetattr(fd,termios.TCSANOW,a)
        print('Jetson 登录：优先使用系统钥匙串；若自动登录失败，请手动输入。密码不会显示或写入日志。\nCtrl+] 取消。',flush=True)
        tty.setraw(0)
        # This is an explicitly opened login terminal, never automatic startup.
        os.write(fd,b'\r')
        recent=b''; password_attempted=False
        while True:
            readable,_,_=select.select([fd,0],[],[],.5)
            if fd in readable:
                chunk=os.read(fd,65536)
                if not chunk: raise RuntimeError('串口断开。')
                os.write(1,chunk);recent=(recent+chunk)[-4096:]
                text=clean_output(recent.decode(errors='replace'))
                if idle_shell_prompt(text):success=True;break
                if not password_attempted and re.search(r'(?:^|\n)[Pp]assword: *$',text):
                    password_attempted=True
                    password=saved_password()
                    if password:
                        for byte in (password+'\r').encode():
                            os.write(fd,bytes([byte]))
                        del password
                        recent=b''
                        continue
                if re.search(r'(?:^|\n)jetson01 login: *$',text):
                    os.write(fd,b'dimos\r');recent=b''
            if 0 in readable:
                chunk=os.read(0,1024)
                if not chunk or b'\x1d' in chunk:break
                os.write(fd,chunk)
    finally:
        termios.tcsetattr(0,termios.TCSANOW,local)
        termios.tcsetattr(fd,termios.TCSANOW,old);os.close(fd);lock.close()
    if success:
        html=urllib.request.urlopen(BASE,timeout=3).read().decode()
        token=re.search(r'name="csrf-token" content="([^"]+)"',html).group(1)
        req=urllib.request.Request(BASE+'/api/start',data=b'',headers={'X-A1Z-Token':token},method='POST')
        urllib.request.urlopen(req,timeout=3).read()
        print('\n已登录，控制台正在自动准备。请回到控制台；机械臂仍需现场确认才启用。')
    else: print('\n已取消登录。')

if __name__=='__main__':
    try:main()
    except Exception as e:print('\n登录未完成：'+str(e));sys.exit(1)
