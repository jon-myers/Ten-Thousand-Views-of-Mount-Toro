(// klank
~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~klanks = File.open(~jsonPath ++ "klank_packets_alt.JSON", "r");
~klanks = ~klanks.readAllString.parseYAML;
~k_freqs = Array.fill(~klanks.size, {|i| [~klanks[i]["freqs"].asFloat]});
~k_amps = Array.fill(~klanks.size, {|i| [~klanks[i]["amps"].asFloat]});
~k_pan = Array.fill(~klanks.size, {|i| ~klanks[i]["pan"].asFloat});

~k_durs = Array.fill(~klanks.size, {|i|
	var dur = ~klanks[i]["rt_dur"];
	dur = if(dur[0].asString=="R", {dur.interpret}, {dur.asFloat});
	dur});

~k_next_durs = Array.fill(~klanks.size - 1, {|i|
	var dur = ~klanks[i+1]["rt_dur"];
	dur = if(dur[0].asString=="R", {dur.interpret}, {dur.asFloat})});
~k_next_durs = ~k_next_durs.add( ~k_durs[~k_durs.size - 1]);

~k_decays = Array.fill(~klanks.size, {|i| [~klanks[i]["rt_decays"]].asFloat});
~k_transient_durs = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_dur"].asFloat});
~k_transient_curve = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_curve"].asFloat});
);

(var server = Server(\nrt,
	options: ServerOptions.new.numOutputBusChannels_(2).numInputBusChannels_(2)
),
defs = Array.fill(7, {|n|
	var name = \klank_ ++ (n+1);
	SynthDef(name, {|out=0, off_dur=2.0, pan=0.0, transient_dur=0.001, transient_curve=0|
		var freqs, decays, amps, sig, sig2, timer, test_dur=2.0, input, input_env;
		// pan.poll;
		freqs = NamedControl.kr(\freqs, 440.0!(n+1));
		decays = NamedControl.kr(\decays, 0.5!(n+1));
		amps = NamedControl.kr(\amps, 0.5!(n+1));
		timer = EnvGen.kr(Env([0,1, 1], [transient_dur, off_dur - transient_dur], transient_curve), doneAction:2);
		input_env = EnvGen.kr(Env([0, 0.01, 0], [transient_dur, off_dur - transient_dur]));
		input = Impulse.ar(0, 0, 0.2);
		sig = Klank.ar(`[freqs, amps, decays], input);
		sig = Pan2.ar((sig) * timer, pan);
		Out.ar(out, sig);
	})
});
defs.do{|def| def.add};
q = Pbind(
	\freqs, Pseq(~k_freqs, 1),
	\decays, Pseq(~k_decays, inf),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq(~k_durs, 1),
	\off_dur, Pseq(16*(~k_next_durs + ~k_durs), 1),
	\pan,Pseq(~k_pan, 1),
	\amps, Pseq(~k_amps, 1),
	\transient_dur, Prand([0.005], inf),
	\transient_curve, Prand([-4, -3, -2, -1, 0, 1, 2, 3, 4], inf)
).asScore(40*60, timeOffset:0.001);

defs.do{|def| q.add([0.0, [\d_recv, def.asBytes]])};
q.sort;

q.recordNRT(
	outputFilePath: "~/nrt_klank.wav".standardizePath,
	sampleRate: 48000,
	headerFormat: "WAV",
	sampleFormat: "int16",
	options: server.options,
	duration: 40 * 60
);
server.remove;
)

