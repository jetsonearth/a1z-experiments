/* Pure routing, shared by the browser and offline tests. */
(function (root) {
  function actionForKey(key, state, repeat, editable) {
    if (repeat || editable || !['positioning','recording','support'].includes(state)) return null;
    if (key === ' ' || key === 'Spacebar') return 'gripper/toggle';
    if (key === 'r' || key === 'R') return state === 'recording' ? 'stop' : 'begin';
    if (key === 'x' || key === 'X') return state === 'support' ? 'discard' : null;
    return null;
  }
  const api = {actionForKey};
  if (typeof module !== 'undefined') module.exports = api;
  else root.captureControls = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
