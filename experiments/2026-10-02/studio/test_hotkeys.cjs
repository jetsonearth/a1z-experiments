const assert = require('node:assert/strict');
const {actionForKey} = require('./static/capture-controls.js');
for (const state of ['positioning','support']) for (const key of ['r','R']) assert.equal(actionForKey(key,state,false,false),'begin');
assert.equal(actionForKey('r','recording',false,false),'stop');
for (const state of ['positioning','support','recording']) {
  assert.equal(actionForKey(' ',state,false,false),'gripper/toggle');
  assert.equal(actionForKey('ArrowLeft',state,false,false),null);
  assert.equal(actionForKey('ArrowRight',state,false,false),null);
  for (const key of [' ','r','R']) {
    assert.equal(actionForKey(key,state,true,false),null);
    assert.equal(actionForKey(key,state,false,true),null);
  }
}
for (const state of ['idle','ready','starting','starting_recording','stopping','finalizing','syncing','error']) {
  for (const key of [' ','r','R']) assert.equal(actionForKey(key,state,false,false),null);
}
for (const key of ['x','X']) {
  assert.equal(actionForKey(key,'support',false,false),'discard');
  assert.equal(actionForKey(key,'support',true,false),null);
  assert.equal(actionForKey(key,'support',false,true),null);
  for (const state of ['positioning','recording','stopping','ready','idle','finalizing']) assert.equal(actionForKey(key,state,false,false),null);
}
console.log('R recording / Space gripper / X discard routing passed, with state/repeat/editable suppression.');
