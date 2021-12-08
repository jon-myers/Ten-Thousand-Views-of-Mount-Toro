(// popcorn
~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~kernals = File.open(~jsonPath ++ "kernals.JSON", "r");
~kernals = ~kernals.readAllString.parseYAML;
~pc_attacks = Array.fill(~kernals.size, {|i| ~kernals[i]["attack"].asFloat});
~pc_hpfs = Array.fill(~kernals.size, {|i| ~kernals[i]["hp_freq"].asFloat});
~pc_lpfs = Array.fill(~kernals.size, {|i| ~kernals[i]["lp_freq"].asFloat});
~pc_decays = Array.fill(~kernals.size, {|i| i.postln; ~kernals[i]["rt_dur"].asFloat - ~pc_attacks[i]});
~pc_durs = Array.fill(~kernals.size-1, {|i| ~kernals[i+1]["rt_onset_time"].asFloat - ~kernals[i]["rt_onset_time"].asFloat});
~pc_durs.add(~pc_durs[~pc_durs.size-1]);
~pc_durs = Array.fill(~pc_durs.size, {|i|
	if (~kernals[i]["type"] == "rest",
		{("Rest(" ++ ~pc_durs[i].asString ++ ")").interpret},
		{~pc_durs[i]})
});
~pc_amps = Array.fill(~kernals.size, {|i| ~kernals[i]["amp"].asFloat});
~pc_pans = Array.fill(~kernals.size, {|i| ~kernals[i]["pan"].asFloat});
);

(
var server = Server(\nrt,
	options: ServerOptions.new.numOutputBusChannels_(2).numInputBusChannels_(2)
),
def = SynthDef(\singlePop, {|out=0, attack=0.01, decay=1.0, curve = -4, lpFilterFreq=1600, hpFilterFreq=50, amp=0.5, pan=0.0|
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

p = Pbind(
	\instrument, \singlePop,
	\attack, Pseq(~pc_attacks, 1),
	\decay, Pseq(~pc_decays, 1),
	\lpFilterFreq, Pseq(~pc_lpfs, 1),
	\hpFilterFreq, Pseq(~pc_hpfs, 1),
	\dur, Pseq(~pc_durs, 1),
	\amp, Pseq(~pc_amps, 1),
	\pan, Pseq(~pc_pans, 1)
).asScore(40 * 60, timeOffset:0.001);

p.add([0.0, [\d_recv, def.asBytes]]);
p.sort;

p.recordNRT(
	outputFilePath: "~/nrt_compressed.wav".standardizePath,
	sampleRate: 48000,
	headerFormat: "WAV",
	sampleFormat: "int16",
	options: server.options,
	duration: 40 * 60
);
server.remove;
)





