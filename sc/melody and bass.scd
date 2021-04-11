(~modes = File.open("/Users/student/Documents/Myers/azure/JSON/modes.JSON", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});

~melody = File.open("/Users/student/Documents/Myers/azure/JSON/melody.JSON", "r");
~melody = ~melody.readAllString.parseYAML.asFloat;


~event_durs = File.open("/Users/student/Documents/Myers/azure/JSON/event_durs.JSON", "r");
~event_durs = ~event_durs.readAllString.parseYAML.asFloat;
);
~triads = Array.fill(~modes.size, {arg i; ~modes[i][0]});
~noc = 15;

~fund = 100;

(
~chords = Pbind(
	\freq, Pseq(~fund * ~triads, ~noc),
	\dur, Pseq(~event_durs, 1),
	\amp, 0.25
);
~mel = Pbind(
	\freq, Pseq(~fund * 2 * ~melody, ~noc),
	\dur, Pseq(~event_durs, 1),
	\amp, 0.5
);
);

(
~chords.play;
~mel.play;
);

~fund * ~triads;

