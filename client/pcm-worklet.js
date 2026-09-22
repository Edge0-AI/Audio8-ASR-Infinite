class Pcm16Worklet extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options.processorOptions || {};
    this.targetSampleRate = processorOptions.targetSampleRate || 16000;
    this.sourceSampleRate = processorOptions.sourceSampleRate || sampleRate;
    this.chunkSamples = processorOptions.chunkSamples || 1600;
    this.resampleRatio = this.sourceSampleRate / this.targetSampleRate;
    this.sourceBuffer = [];
    this.sourcePosition = 0;
    this.output = new Int16Array(this.chunkSamples);
    this.outputIndex = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const channel = input[0];
    for (let i = 0; i < channel.length; i += 1) {
      this.sourceBuffer.push(channel[i]);
    }

    while (this.sourcePosition + 1 < this.sourceBuffer.length) {
      const index = Math.floor(this.sourcePosition);
      const fraction = this.sourcePosition - index;
      const left = this.sourceBuffer[index] || 0;
      const right = this.sourceBuffer[index + 1] || left;
      const sample = left + (right - left) * fraction;
      const clipped = Math.max(-1, Math.min(1, sample));
      this.output[this.outputIndex] =
        clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff;
      this.outputIndex += 1;
      this.sourcePosition += this.resampleRatio;

      if (this.outputIndex >= this.output.length) {
        const packet = this.output.slice();
        this.port.postMessage(packet.buffer, [packet.buffer]);
        this.output = new Int16Array(this.chunkSamples);
        this.outputIndex = 0;
      }
    }

    const consumed = Math.floor(this.sourcePosition);
    if (consumed > 0) {
      this.sourceBuffer.splice(0, consumed);
      this.sourcePosition -= consumed;
    }

    return true;
  }
}

registerProcessor("pcm16-worklet", Pcm16Worklet);

