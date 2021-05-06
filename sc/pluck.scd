~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/'

// ~modes = File.open("/Users/student/Documents/Myers/azure/JSON/modes_and_variations.JSON", "r");
~modes = File.open(~jsonPath ++ "modes_and_variations.JSON", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});
~var0 = ~modes[1];
~var1 = ~modes[2];
~modes= ~modes[0];

~triads = Array.fill(~modes.size, {arg i;
	var pitches;
	pitches = ~modes[i][..2];
	pitches[1] = pitches[1] * 2;
	pitches;
});

~alt_triads_0 = Array.fill(~modes.size, {arg i;
	var pitches;
	pitches = ~var0[i][..2];
	pitches[1] = pitches[1] * 2;
	pitches;
});

~alt_triads_1 = Array.fill(~modes.size, {arg i;
	var pitches;
	pitches = ~var1[i][..2];
	pitches[1] = pitches[1] * 2;
	pitches;
});

(
~triad_fixer = {arg triad = [100, 100, 100], min;
	var dub_min = 2 * min;
	var fixed_triad = Array.fill(triad.size, {|i|
		if(triad[i] >= dub_min, {triad[i]/ ( 2 ** log2(triad[i]/min).floor)}, {triad[i]});
	});
	fixed_triad = Array.fill(triad.size, {|i|
		if(fixed_triad[i] < min, {fixed_triad[i]/ ( 2 ** log2(fixed_triad[i]/min).floor)}, {fixed_triad[i]});
	});

	fixed_triad = fixed_triad.sort;
	if(fixed_triad[1] / fixed_triad[0] < 1.4, {fixed_triad[1] = fixed_triad[1] * 2});
	fixed_triad = fixed_triad.sort;
	fixed_triad.postln
};
)

~pluck = {|freq, coef| Pluck.ar(Pulse.ar(freq) * 0.1, Impulse.ar(0), 1/freq, 1/freq, 5, coef)}


(
SynthDef.new('pluck', {arg coef=0.1;
	var p0, p1, p2, sig, line, buf, freqs;
	freqs = \freqs.kr(200!3);
	line = Line.kr(0.2, 0, 5, doneAction: 2);
	p0 = ~pluck.value(freqs[0], coef);
	p1 = ~pluck.value(freqs[1], coef);
	p2 = ~pluck.value(freqs[2], coef);
	sig = p0 + p1 + p2;
	Out.ar(0, Pan2.ar(sig));
}).add
):

x = Synth('pluck', [\freqs, [200, 300, 500]]);

(
a = Pbind(
	\instrument, \pluck,
	\freqs, ~triad,
	\dur, Prand([1, 1.2, 1.4], inf),
	\coef, ~coef
).play);
a.stop;

~coef = PatternProxy(0.1);
~coef.source = 0.2;
~triad = PatternProxy(~triad_fixer.value(200 * ~triads[], 150));
~triad.source = [~triad_fixer.value(200 * ~triads[8], 100)];
~triad_fixer.value(200 * ~triads[0], 150)




~triad_fixer.value([200.0, 300.0, 500.0], 100.0)

