~modes = File.open("/Users/jon/Documents/2021/azure/modes.json", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});
~fund = 150;

(
SynthDef.new('tri', {arg freq, dur, amp, release=0.1, gate = 1; var sig, env, gen;
	sig = LFTri.ar(freq);
	env = Env.asr(0.2, amp, 0.5, -2);
	gen = EnvGen.kr(env, gate, doneAction: Done.freeSelf);
	Out.ar(0, [1, 1] * sig * gen)
}).add
);

~get_sub_mode = {arg mode, to_add; var sub_mode, given, choices, selection, selections;
	given = [0, 2, 4];
	choices = [1, 3, 5, 6];
	selection = choices.choose;
	choices.remove(selection);
	selections = [selection, choices.choose];
	choices.remove(selection);
	selections = if(to_add == 3, {selections ++ [selections.choose]}, {selections});
	given = given ++ choices;
	given = given.sort;
	sub_mode = Array.fill(5, {arg i; mode[given[i]]});
};

~mode = ~get_sub_mode.value(~modes[10]).sort
~mode = (0.5 * ~mode) ++ ~mode ++ (2 * ~mode) ++ (4 * ~mode)


(
(
(
~pattern = File.open("/Users/jon/Documents/2021/azure/pattern.json", "r");
~pattern = ~pattern.readAllString.parseYAML;
~pattern = Array.fill(~pattern.size, {arg i; ~pattern[i].asFloat});

~durs = File.open("/Users/jon/Documents/2021/azure/durs.json", "r");
~durs = ~durs.readAllString.parseYAML[0];
~durs = Array.fill(~durs.size, {arg i; ~durs[i].asFloat});
~durs[~durs.size-1] = ~durs[~durs.size-1] * 1.9;
~durs[~durs.size-2] = ~durs[~durs.size-2] * 1.3;


);
	(
~freq_pattern = Array.fill(~pattern.size, {arg i; var dyad;
	dyad = ~fund * [~mode[5 + ~pattern[i][0]], ~mode[5 + ~pattern[i][1]]];
}));
);

((
a = Pbind(
	\instrument, 'tri',
	\freq, Pseq(~freq_pattern, 1),
	\dur, Pseq(~durs, ~freq_pattern.size),
));
a.play;
)
)

Env.adsr(0.1, 0.1, 5/7, 0.5, 1, -1).test(5).plot;

[1, 2, ] ++ [3, 4, ]