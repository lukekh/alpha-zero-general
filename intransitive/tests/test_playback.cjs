const {test} = require('node:test');
const assert = require('node:assert/strict');
const {MovePlayback} = require('../playback.js');

test('producer can queue arbitrarily many moves without waiting for UI clock', () => {
  let now = 0;
  const player = new MovePlayback(() => now);
  player.reset([{ply: 0}]);
  for (let ply = 1; ply <= 20; ply++) player.push({ply});
  assert.equal(player.latest.ply, 20);
  assert.equal(player.visible.ply, 0);
  now = 999;
  assert.equal(player.tick(), false);
  now = 1000;
  assert.equal(player.tick(), true);
  assert.equal(player.visible.ply, 1);
  assert.equal(player.tick(), false);
  now = 2000;
  player.tick();
  assert.equal(player.visible.ply, 2);
});

test('slow searches add no artificial delay, while zero interval is immediate', () => {
  let now = 0;
  const player = new MovePlayback(() => now);
  player.reset([{ply: 0}]);
  now = 5000;
  player.push({ply: 1});
  assert.equal(player.tick(), true);
  player.interval = 0;
  player.push({ply: 2});
  assert.equal(player.tick(), true);
});

test('review pauses only playback; latest and restart discard no unintended moves', () => {
  let now = 0;
  const player = new MovePlayback(() => now);
  player.reset([{ply: 0}, {ply: 1}, {ply: 2}]);
  player.seek(0);
  player.push({ply: 3});
  now = 5000;
  assert.equal(player.tick(), false);
  assert.equal(player.visible.ply, 0);
  assert.equal(player.latest.ply, 3);
  player.seek(1);
  assert.equal(player.visible.ply, 1);
  player.seek(100, true);
  assert.equal(player.visible.ply, 3);
  assert.equal(player.playing, true);
  player.seek(-100);
  assert.equal(player.visible.ply, 0);
  player.reset([{ply: 0}]);
  now = 10000;
  assert.equal(player.tick(), false);
  assert.equal(player.frames.length, 1);
});
