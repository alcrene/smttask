from tasks import GenerateTask, ProcessTask
from parameters import ParameterSet

paramfile = 'params/gen.ntparameterset'
params = ParameterSet(paramfile)
    # τ=1, σ=1, seed=0
x1 = GenerateTask(paramfile)
x2 = GenerateTask(params)
x3 = GenerateTask(τ=1, σ=1, seed=1)

y1  = ProcessTask(x1)
y1b = ProcessTask(x1)  # Does not reexecute x1 or ProcessTask
signal = x2.run(cache=True)      # Force immediate execution of x2
    # Because it has the same signature, x1 will find its result
    # on disk and load it
y2  = ProcessTask(x2)  # Will not reexecute x2
y2b = ProcessTask(signal)  # **Does** reexcute ProcessTask, because input is different
y3  = ProcessTask(x3)
x3.run()               # Still preexecutes x3, since y3 was not run yet
    # x3 has a different signature to x1 and x2, and therefore will run

for y in [y1, y2, y2b, y3]:
    y.execute()
    # Execute order:
    # x1 (load), y1, y1b (load), y2, y2b, x3 (load), y3
    #
    # Remarks:
    # - y1b just need the signature of x1 to find its result, and therefore
    #   x1 is not loaded a second time
    # - We did not cache x3, so it still has to be loaded
