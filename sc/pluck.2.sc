
(
~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~plucks = File.open(~jsonPath ++ "all_plucks.JSON", "r");
~plucks = ~plucks.readAllString.parseYAML;
~freqs = Array.fill(~plucks["freqs"].size, {arg i; var obj;
	obj = ~plucks["freqs"][i];
	if(obj == "Rest()", {Rest()}, {[obj.asFloat]})
});
~dur = ~plucks["rt_dur"].asFloat;
~coef = ~plucks["coef"].asFloat;
~decay = ~plucks["decay"].asFloat;
~delays = Array.fill(~plucks["delays"].size, {arg i; [~plucks["delays"][i].asFloat]});
~vols = Array.fill(~plucks["vol"].size, {arg i; [~plucks["vol"][i].asFloat]});

~pluck = {|freq, coef, decay| Pluck.ar(Pulse.ar(freq) * 0.1, Impulse.ar(0), 1/freq, 1/freq, decay, coef)};
);


~klank_times = Array.fill(3, {3.0.rand});
(
SynthDef.new('pluck', {arg coef=0.1, decay=3;
	var p0, p1, p2, sig, line, buf, freqs, delays, vols, formlet;
	delays = \delays.kr(0.5!3);
	freqs = \freqs.kr(200!3);
	vols = \vols.kr(0.5!3);
	line = Line.kr(0.2, 0, 2*decay, doneAction: 2);
	p0 = ~pluck.value(freqs[0], coef, decay);
	p1 = ~pluck.value(freqs[1], coef, decay);
	p2 = ~pluck.value(freqs[2], coef, decay);
	p0 = AllpassN.ar(p0, delays[0], delays[0], 0, vols[0]);
	p1 = AllpassN.ar(p1, delays[1], delays[1], 0, vols[1]);
	p2 = AllpassN.ar(p2, delays[2], delays[2], 0, vols[2]);

	sig = p0 + p1 + p2;
	// sig = sig + formlet;
	Out.ar(0, Pan2.ar(sig));
	// Out.ar(~test_bus, sig);
}).add
);
~klang_times
~amp = nil;
{Klank.ar(`[[200, 300, 500], ~amp, ~klang_times], Impulse.ar(0, 0, 0.2))}.play;
{Klank.ar(`[~freqs, ~amps, ~ring_times], Impulse.ar(0, 0, 0.2), ~scale_factor)}.play;
z = Synth('symp', ['freq', 300]);

(
a = Pbind(
	\instrument, \pluck,
	\freqs, Pseq(~freqs, 1),
	\dur, Pseq(~dur, 1),
	\coef, Pseq(~coef, 1),
	\decay, Pseq(~decay, 1),
	\delays, Pseq(~delays, 1),
	\vols, Pseq(~vols, 1)
).play)

~plucks["vol"]

Record