~fund = 100;
~trip = {arg a; a ++ (2*a) ++ (4*a)};

~registrate = {arg a; var first_oct, second_oct, third_oct, out;
	a.postln;
	first_oct = a[..2];
	second_oct = a[2..4];
	third_oct = a[4..6];
	first_oct = first_oct ++ (2*first_oct) ++ (4 * first_oct);
	first_oct.postln;
	second_oct = 2*second_oct ++ 4 * second_oct;
	third_oct = 4 * third_oct;
	out = first_oct ++ second_oct ++ third_oct;
}


~registrate.value(~modes[0])



~modes = File.open("/Users/jon/Documents/2021/azure/modes.json", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});
~modes.readAllString



(~dc_alg = {arg choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0;
	var selections = [], weight_array, sum, probs;
	if (weights ==0) {weights = Array.fill(choices.size, {1})};
	if (counts == 0) {counts = Array.fill(choices.size, {1})};
	epochs.do({var selection_index;
		weight_array = Array.fill(choices.size, {arg i; weights[i] * (counts[i]**alpha)});
		probs = Array.fill(choices.size, {arg i; weights[i] * (counts[i]**alpha) / weight_array.sum});
		selection_index = Array.series(choices.size).wchoose(probs);
		counts = Array.fill(counts.size, {arg i; counts[i] + 1}).put(selection_index,0);
		selections = selections ++ [choices[selection_index]];
	});
	selections;
});

~t_0 =[1, 1.2222222222222223, 1.5, 1.8333333333333333, 1.1666666666666667, 1.4, 1.6923076923076923]
(
p = Pbind(
	\freq, Pseq(~dc_alg.value(~trip.value(~t_0) * ~fund, 1000), inf),
	\dur, Prand( [0.05, 0.08, 0.09, 0.1,0.15,0.2,0.25,0.3], inf),
	\vol, Prand([0.25, 0.4, 0.65], inf)
).play;
);
// {Pan2.ar(SinOsc.ar(~fund * ~modes[0][0], 0, 0.0125))}.play;
)

a = [0, 1, 2, 3, 4]
a[1..2]