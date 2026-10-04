"""Shared serial shell handshake. Never interrupts an unknown foreground task."""
import os
import re
import select
import time

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')

def clean_output(text):
    return ANSI.sub('', text).replace('\r', '').replace('\x00', '')

def idle_shell_prompt(text):
    return bool(re.search(r'(?:^|\n)[ \t]*dimos@jetson01[^\n]*[\$%] $', clean_output(text)))

def ensure_shell(fd):
    # Only redraw. Enter is sent only AFTER an exact idle-shell observation.
    os.write(fd, b'\x0c')
    observed = b''
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if select.select([fd], [], [], .1)[0]:
            chunk = os.read(fd, 65536)
            if not chunk:
                raise RuntimeError('Jetson 串口已断开；重新接好后点重新准备。')
            observed = (observed + chunk)[-8192:]
            clean = clean_output(observed.decode(errors='replace'))
            if idle_shell_prompt(clean):
                break
    else:
        clean = clean_output(observed.decode(errors='replace'))
        if re.search(r'(?:login:|[Pp]assword:)\s*$', clean):
            raise RuntimeError('Jetson 需要登录。当前 USB-C 仅提供串口；登录完成后点重新准备，后续检查自动完成。')
        raise RuntimeError('Jetson idle shell not confirmed; no recording command sent')
    if clean.endswith('dimos@jetson01 A1Z $ '):
        return
    # Pace the short bootstrap too: zsh syntax highlighting can overwhelm this
    # serial console even before the larger RPC payload. Never exec the login shell.
    def send(command):
        for byte in command:
            os.write(fd, bytes([byte]))
            time.sleep(.025)
    def wait_prompt(pattern):
        data = b''
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if select.select([fd], [], [], .1)[0]:
                chunk = os.read(fd, 65536)
                if not chunk: break
                data = (data + chunk)[-8192:]
                text = clean_output(data.decode(errors='replace'))
                if re.search(pattern, text): return
        raise RuntimeError('Jetson 命令行初始化未完成；未发送采集命令。')
    send(b'bash --noprofile --norc\r')
    wait_prompt(r'(?:^|\n)bash-[0-9.]+\$ $')
    send(b"PS1='dimos@jetson01 A1Z $ '\r")
    wait_prompt(r'(?:^|\n)dimos@jetson01 A1Z \$ $')
