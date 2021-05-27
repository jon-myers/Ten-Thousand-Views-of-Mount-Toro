(~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~klanks = File.open(~jsonPath ++ "klank_packets.JSON", "r");
~klanks = ~klanks.readAllString.parseYAML;

~freqs = Array.fill(~klanks.size, {|i| [~klanks[i]["freqs"].asFloat]});
~amps = Array.fill(~klanks.size, {|i| [~klanks[i]["amps"].asFloat]});
~pan = Array.fill(~klanks.size, {|i| ~klanks[i]["pan"].asFloat});

~durs = Array.fill(~klanks.size, {|i|
	if(~klanks[i]["type"] == "rest",
		{Rest(~klanks[i]["rt_dur"].asFloat)},
		{~klanks[i]["rt_dur"].asFloat}
)});

~decays = Array.fill(~klanks.size, {|i| [~klanks[i]["rt_decays"]].asFloat});

~transient_durs = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_dur"].asFloat});
~transient_curve = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_curve"].asFloat});
)

~klanks[0]["transient_dur"]
// SynthDef from 'factory'
(
(1..7).do{|n|
	var name = \klank_ ++ n;
	SynthDef(name, {|out=0, off_dur=2.0, pan=0.0, transient_dur=0.001, transient_curve=0|
		var freqs, decays, amps, sig, timer, test_dur=2.0;
		// pan.poll;
		freqs = NamedControl.kr(\freqs, 440.0!n);
		decays = NamedControl.kr(\decays, 0.5!n);
		amps = NamedControl.kr(\amps, 0.5!n);
		timer = EnvGen.kr(Env([0,1, 1], [transient_dur, off_dur - transient_dur], transient_curve), doneAction:2);
		sig = Klank.ar(`[freqs, amps, decays], Impulse.ar(0, 0, 0.2));
		sig = Pan2.ar(sig * timer, pan);
		Out.ar(out, sig);
	}).add;
}
);

/*~freqs = [[[200, 300, 500]], [[250, 350, 450]], [[200, 450, 510]]];
~decays = [[[0.5, 1.3, 1.0]], [[0.75, 2.1, 0.3]], [[0.7, 0.8, 0.3]]];*/
// ~pan = Array.fill(~freqs.size, {2.0.rand - 1});
(
p = Pbind(
	\freqs, Pseq(~freqs, 1),
	\decays, Pseq(~decays, inf),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq(~durs, 1),
	\off_dur, Pkey(\decays).collect({|x| 8*x[0].maxItem.asSymbol}),
	\pan,Pseq(~pan, 1),
	\amps, Pseq(~amps, 1),
	\transient_dur, Prand([0.005], inf),
	\transient_curve, Prand([-4, -3, -2, -1, 0, 1, 2, 3, 4], inf)
).play
);



~freqs[0][0][0] 