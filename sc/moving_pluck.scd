{Pluck.ar(Impulse.ar(0), Impulse.kr(0), 600.reciprocal, (300 * (2 ** SinOsc.kr(1, 0))).reciprocal, 5.0, 0.3, 0.5)}.play

Env.new([200, 800], [1], 'exponential').plot;

(
~freqEnv = {|spec, dur|
	var start, pivot, target, center_0, center_1, width_0, width_1, centerEnv, widthEnv, freqEnv, trigEnv;
	start = spec[0];
	pivot = spec[1];
	target = spec[2];
	center_0 = (pivot + start) / 2;
	center_1 = (pivot + target) / 2;
	centerEnv = EnvGen.kr(Env.new([center_0, center_1], [dur], 'exponential'));
	width_0 = (start/pivot).log2.abs;
	width_1 = (target/pivot).log2.abs;
	widthEnv = EnvGen.kr(Env.new([width_0, width_1], [dur], 'lin'));
	freqEnv = centerEnv * (2 ** SinOsc.kr(dur.reciprocal, 0.75, widthEnv)).poll;
	trigEnv = EnvGen.kr(Env.new([1, 1, 0], [dur/2, 0]));
	freqEnv = Gate.kr(freqEnv, trigEnv);
}
);

Gate

(
SynthDef.new(\moving_pluck, {|out=0|
	var spec, freqEnv, sig, transientEnv;
	spec = NamedControl.kr(\spec, 200 * [3/2, (4/3) * (3/2), (5/4)]);
	freqEnv = ~freqEnv.value(spec, 1.0);
	transientEnv = EnvGen.kr(Env.new([0, 1], [0.05]));
	sig = Pluck.ar(Impulse.ar(0), Impulse.kr(0), ~pivot.reciprocal, freqEnv.reciprocal, 3, 0.2, 0.5);
	sig = sig * transientEnv;
	Out.ar(out, sig);
}).add;
)



Synth(\moving_pluck)


~freqEnv.value([~start, ~pivot, ~target], 1.0);

~start = 3/2;
~pivot = (4/3) * (3/2);
~target = (5/4);
~start * 200
~target * 200
{SinOsc.ar(~target*200, 0, 0.0125)}.play
{SinOsc.ar(~start*200, 0, 0.0125)}.play
~center_0 = (pivot + start) / 2;
~width_0 = (~start/~pivot).log2.abs