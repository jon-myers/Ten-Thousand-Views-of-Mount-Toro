~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/'

// ~modes = File.open("/Users/student/Documents/Myers/azure/JSON/modes_and_variations.JSON", "r");
~modes = File.open(~jsonPath ++ "modes_and_variations.JSON", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});
~var0 = ~modes[1];
~var1 = ~modes[2];
~modes= ~modes[0];

~generated_list = File.open(~jsonPath ++ "triads.JSON", "r")
~generated_list = ~generated_list.readAllString.parseYAML;
~generated_list = Array.fill(~generated_list.size, {arg i; ~generated_list[i].asFloat});


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

x = Synth('pluck', [\freqs, [200, 300, 500], \delays, [0, 0.1, 0.2]]);

(
a = Pbind(
	\instrument, \pluck,
	\freqs, Pseq([Rest()] ++ ~generated_list[2..5], inf),
	\dur, Pseq([1.5, 3, 3, 3], inf),
	\coef, ~coef,
	\decay, ~decay,
	\delays, [[0.0, 0.0, 0.0]]
).play);
a.stop;
[[[Rest(), Rest(), Rest()]]] ++ ~generated_list[2..5]
[[0, 0, 0]] ++ [[1, 0, 0]]

[[[Rest(), Rest(), Rest()]]] ++ ~generated_list[2..5]

~list = PatternProxy(Pseq(~generated_list, inf));
~list.source = Pseq(~generated_list[0], inf);
~chord.source
~chord = PatternProxy(~triad_fixer.value(~triads[0], 100));
~chord.source = [~triad_fixer.value(~triads[10], 100)];
~coef = PatternProxy(0.1);
~coef.source = 0.91;
~decay = PatternProxy(3);
~decay.source = 10;
90*1.6

~generated_list

~generated_list[2..5]

[Rest()] ++ ~generated_list[2..5]
0.65 * (2 ** 0.5)