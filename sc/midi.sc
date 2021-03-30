a = SimpleMIDIFile.read("/Users/student/Documents/Myers/azure/test_midi.MIDI")

(
p = "/Users/student/Documents/Myers/azure/reaper/samples/drum_sample_0.wav";
~buffers = Array.fill(5, {arg i;
	p = "/Users/student/Documents/Myers/azure/reaper/samples/drum_sample_"++ i ++ ".wav";
	Buffer.read(s, p);
}));

(
SynthDef('sampler', {|midinote = 0|
	var sig, bufnum;
	bufnum = ~buffers[0].bufnum + midinote;
	SendTrig.kr(Impulse.kr(4), 0, bufnum);
	sig = PlayBuf.ar(2, bufnum);
	Out.ar(0, sig)}).add;
);


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



~fund = 150;




(
// ~modes = File.open("/Users/student/Documents/Myers/azure/modes.json", "r");
~modes = File.open("/Users/jon/Documents/2021/azure/modes.json", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});
~triads = Array.fill(~modes.size, {arg i; ~modes[i][..2]});
~quartet = Array.fill(~modes.size, {arg i; var mode;
	mode = ~modes[i][..3];
	mode[0] = mode[0]/2;
	mode[1] = mode[1]*2;
});

~quintet = Array.fill(~modes.size, {arg i; var mode;
	mode = ~modes[i][..4];
	mode[0] = mode[0]/2;
	mode[1] = mode[1]*2;
	mode[3] = mode[3] * 2;

});
);

~divs = Array.fill(~modes.size, {[3, 4, 5, 6, 7, 8, 9, 10, 11].choose});
~melody_durs = Array.fill(~divs.size, {arg i;
	Array.fill(~divs[i], {1/~divs[i]})
}).flat;

(
~chords = Pbind(
	\freq, Pseq(~fund * ~quartet, 1),
	\dur, 1,
	\amp, 0.1
);
~melody = Pbind(
	\freq, Pseq(Array.fill(~modes.size, {arg i; Pseq(~dc_alg.value(~fund * 2 * ~modes[i], ~divs[i]), 1)})),
	\dur, Pseq(~melody_durs, 1)
));

(
~chords.play;
~melody.play;
)


