~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';

~packets = File.open(~jsonPath ++ "packets.JSON", "r");
~packets = ~packets.readAllString.parseYAML;
~packets.do({arg i; i[\freqs].postln})
Array
~packets[2].values

d = Dictionary.with(*[\a->1,\b->2,\c->3])
~packets[0][\freqs]
d
~packets[0][\c]


~pluck = {|freq, coef, decay| Pluck.ar(Pulse.ar(freq) * 0.1, Impulse.ar(0), 1/freq, 1/freq, decay, coef)};

(
SynthDef.new('pluck', {arg coef=0.1, decay=3;
	var p0, p1, p2, sig, line, buf, freqs, delays;
	delays = \delays.kr(0.0!3);
	freqs = \freqs.kr(200!3);
	line = Line.kr(0.2, 0, 2*decay, doneAction: 2);
	p0 = ~pluck.value(freqs[0], coef, decay);
	p1 = ~pluck.value(freqs[1], coef, decay);
	p2 = ~pluck.value(freqs[2], coef, decay);
	p0 = AllpassN.ar(p0, 1.0, delays[0], 0);
	p1 = AllpassN.ar(p1, 1.0, delays[1], 0);
	p2 = AllpassN.ar(p2, 1.0, delays[2], 0);
	sig = p0 + p1 + p2;
	Out.ar(0, Pan2.ar(sig));
}).add
);

