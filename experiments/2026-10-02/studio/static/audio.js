/* Local Kokoro clips: generated once, played only after remote acknowledgment. */
(function () {
  const clips = {'已就绪':'ready','开始录制':'recording','本条已结束，可以复位':'ended',
    '夹爪已张开':'opening','夹爪已闭合':'closing','夹爪张开指令已发送':'opening','夹爪闭合指令已发送':'closing'};
  const players = Object.fromEntries([...new Set(Object.values(clips))].map(name => {
    const audio = new Audio('/audio/' + name + '.wav'); audio.preload='auto';
    return [name,audio];
  }));
  let active=null;
  window.playAcknowledgment = function (event) {
    const name=clips[event.text]; if(!name)return;
    if(active){active.pause();active.currentTime=0;}
    active=players[name];active.currentTime=0;
    active.play().catch(() => {}); // Autoplay unlocks after the first click/keypress on the page.
  };
})();
