(
SynthDef.new(\popcorn, {|out=0, density=1|
	var sig, env, envGen, rands, trig, randVal;
	rands = Array.fill(1000, {1.0.rand});
	trig = Dust.kr(density);
	randVal = Select.kr(Stepper.kr(trig, 0, 0, 1000), rands).poll;
	env = Env.perc(0.01, 0.125 * (2 ** (4 * randVal)));
	envGen = EnvGen.kr(env, trig);
	sig = envGen * PinkNoise.ar(0.2);
	Out.ar(out, sig);
}).add;
)

x = Synth(\popcorn, [\density, 2]);

{ Stepper.kr(Impulse.kr(100), 0,1,10,1) }.plot(0.3,minval:0,maxval:10);
