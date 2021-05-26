~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~klanks = File.open(~jsonPath ++ "klank_packets.JSON", "r");
~klanks = ~klanks.readAllString.parseYAML;

~freqs = Array.fill(~klanks.size, {|i| [~klanks[i]["freqs"].asFloat]});

~durs = Array.fill(~klanks.size, {|i|
	if(~klanks[i]["type"] == "rest",
		{Rest(~klanks[i]["rt_dur"].asFloat)},
		{~klanks[i]["rt_dur"].asFloat}
)});

~decays = Array.fill(~klanks.size, {|i| [~klanks[i]["rt_decays"]].asFloat});


~klanks[1]["rt_decays"]







(
~dels = Array.fill(2, {2.0});
{ [1, 1] * Klank.ar(`[1*[100, 150], nil, ~dels], Impulse.ar(0, 0, 0.1)) }.play;
)


// SynthDef from 'factory'
(
(1..7).do{|n|
	var name = \klank_ ++ n;
	SynthDef(name, {|out=0, off_dur=2.0, pan=0.0|
		var freqs, decays, sig, timer, test_dur=2.0;
		freqs = NamedControl.kr(\freqs, 440.0!n);
		decays = NamedControl.kr(\decays, 0.5!n);
		timer = EnvGen.kr(Env([0,1, 1], [0.001, off_dur -0.001]), doneAction:2);
		sig = Klank.ar(`[freqs, nil, decays], Impulse.ar(0, 0, 0.2));
		sig = Pan2.ar(sig * timer, pan);
		Out.ar(out, sig);
	}).add;
}
);

~freqs = [[[200, 300, 500]], [[250, 350, 450]], [[200, 450, 510]]];
~decays = [[[0.5, 1.3, 1.0]], [[0.75, 2.1, 0.3]], [[0.7, 0.8, 0.3]]];
~pan = Array.fill(~freqs.size, {2.0.rand - 1});
(
p = Pbind(
	\freqs, Pseq(~freqs, 1),
	\decays, Pseq(~decays, inf),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq(~durs, 1),
	\off_dur, Pkey(\decays).collect({|x| x[0].maxItem.asSymbol}),
	\pan,Pseq(~pan, 1)
).play
);



~freqs[0][0][0] 