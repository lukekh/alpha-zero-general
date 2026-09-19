/* UI-only clock. No network requests or engine state are controlled here. */
class MovePlayback {
  constructor(now = () => performance.now()) {
    this.now = now;
    this.frames = [];
    this.index = 0;
    this.playing = true;
    this.interval = 1000;
    this.shownAt = now();
  }
  reset(frames) {
    this.frames = [...frames];
    this.index = this.frames.length - 1;
    this.playing = true;
    this.shownAt = this.now();
  }
  get visible() { return this.frames[this.index]; }
  get latest() { return this.frames.at(-1); }
  get atLatest() { return this.index === this.frames.length - 1; }
  push(frame) { this.frames.push(frame); }
  seek(index, follow = false) {
    this.index = Math.max(0, Math.min(index, this.frames.length - 1));
    this.playing = follow;
    this.shownAt = this.now();
  }
  tick() {
    if (!this.playing || this.atLatest || this.now() - this.shownAt < this.interval) return false;
    this.index++;
    this.shownAt = this.now();
    return true;
  }
}
if (typeof module !== 'undefined') module.exports = {MovePlayback};
