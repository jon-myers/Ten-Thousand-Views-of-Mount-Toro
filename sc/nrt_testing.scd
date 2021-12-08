// (
// x = Pbind(
// 	\freq, Pexprand(200, 800, inf),
// 	\dur, Pexprand(0.8, 1.25, inf) * Pgeom(0.01, 1.0143978590819, 400),
// 	\legato, 3,
// ).asScore(10, timeOffset: 0.001);
//
// x.add([0.0, [\d_recv, SynthDescLib.global[\default].def.asBytes]]);
// x.sort;
//
// x.recordNRT(
// 	outputFilePath: "~/nrt.aiff".standardizePath,
// 	sampleRate: 44100,
// 	headerFormat: "AIFF",
// 	sampleFormat: "int16",
// 	options: ServerOptions.new.numOutputBusChannels_(2),
// 	duration: 10
// );
// )









(var server = Server(\nrt,
	options: ServerOptions.new
	.numOutputBusChannels_(2)
	.numInputBusChannels_(2)
),
 def = SynthDef.new(\singlePop, {|out=0, attack=0.01, decay=1.0, curve=-4, lpFilterFreq=1600, hpFilterFreq=50, amp=0.5, pan=0.0|
	var sig, env, envGen;
	env = Env.perc(attack, decay);
	envGen = EnvGen.kr(env, 1.0, doneAction:2);
	sig = PinkNoise.ar();
	sig = LPF.ar(sig, lpFilterFreq);
	sig = HPF.ar(sig, hpFilterFreq);
	sig = envGen * sig;
	sig = 2 * sig * amp;
	sig = Pan2.ar(sig, pan);
	Out.ar(out, sig);
});
def.add;







(
SynthDef.new(\singlePop, {|out=0, attack=0.01, decay=1.0, curve = -4, lpFilterFreq=1600, hpFilterFreq=50, amp=0.5, pan=0.0|
	var sig, env, envGen;
	env = Env.perc(attack, decay);
	envGen = EnvGen.kr(env, 1.0, doneAction:2);
	sig = PinkNoise.ar();
	sig = LPF.ar(sig, lpFilterFreq);
	sig = HPF.ar(sig, hpFilterFreq);
	sig = envGen * sig;
	sig = 2 * sig * amp;
	sig = Pan2.ar(sig, pan);
	Out.ar(out, sig);
}).add
);