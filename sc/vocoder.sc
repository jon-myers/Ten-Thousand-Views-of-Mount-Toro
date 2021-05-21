

SynthDef.new('vocoder', {arg freq;
	var sig, amp, vocoder;
	sig = LFTri.ar(200 * (2 ** Lag.kr(MouseX.kr(0, 1), 0.1)));
	amp = Amplitude.kr(BPF.ar(sig, 300));
	vocoder = SinOsc.ar(300, 0, amp) ;

	Out.ar(0, [sig * 0.4, vocoder]);
}).add

z = Synth('vocoder')