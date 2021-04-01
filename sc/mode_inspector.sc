/*~modes = File.open("/Users/jon/Documents/2021/azure/modes.json", "r");*/
~modes = File.open("/Users/student/Documents/Myers/azure/modes.json", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});


~mode = ~modes[12]
~mode = ~mode.sort;
~fund = 100;
~degree_sieve = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6];

(
var notes, on, off;
MIDIClient.init;
MIDIIn.connectAll;
notes = Array.newClear(128);

on = MIDIFunc.noteOn({ arg vel, num; var degree, oct;
	degree = num % 12;
	oct = (num / 12).floor;
	notes[num] = Synth(\default, [\freq, ~mode[~degree_sieve[degree]] * (2 ** oct) * ~fund, \amp,
		vel * 0.00315]);
});

off = MIDIFunc.noteOff({ |vel, num|
	notes[num].release;
});


q = {on.free; off.free;}

)

// when finished
q.free;