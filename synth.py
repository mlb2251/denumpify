import numpy as np
import mlb
from util import *
from tqdm import tqdm
from stack_program import *
from compiletime import pre_synth
from queue import PriorityQueue


def synthesize(fns,cfg):
    priors,consts,vars,fns = pre_synth(fns,cfg)

    steps = []
    for f in fns:
        argcs = f.argcs if f.argcs is not None else [0,1,2,3]
        for argc in argcs:
            steps.append(FuncStep(f, priors.get(f.name,priors['_default']), argc))

    steps += [ConstStep(x, priors['_const'], str(x)) for x in consts]
    steps += [VarStep(x, priors['_var'], str(x)) for x in vars]

    steps.sort(key=lambda step: step.prior)

    env = {
        'x': lambda: np.ones((2,3)),
        'y': lambda: np.array([[1,2,3],[4,5,6]]),
    }
    search_state = SearchState(
                            steps,
                            n=3,
                            env=env,
                            )

    p = StackProgram(
                p=(),
                search_state=search_state,
                ll=0,
                stack_state=StackState(())
                )

    heap = PriorityQueue()
    heap.put(p.heapify())


    print = tqdm.write
    for _ in tqdm(range(100), disable=True):
        base_p = heap.get().p
        if base_p.deleted:
            continue
        print(f'expanding {base_p.pretty()}')
        ps = base_p.expand()
        if len(ps) == p.search_state.n:
            heap.put(base_p.heapify()) # return it to the heap bc there might be more expansions
        for p in ps:
            if len(p.p) < search_state.max_program_size:
                heap.put(p.heapify()) # only use as a potential parent program if its small enough
            print(f'\t{p.pretty()}')
    
    results = [p for p in search_state.seen_stack_states.values() if len(p.stack_state) == 1]
    results.sort(key=lambda p: -p.ll)
    for res in reversed(results):
        print(res.pretty())
    
    # quick test of the parser
    StackProgram.parse('(0, mean.1)',search_state)

    repl(search_state)

    print("done")


def repl(search_state):
    import readline
    seen = search_state.seen_stack_states
    str_seen = {str(p):p for p in seen.values()}
    while True:
        try:
            line = input('>>> ').strip()
        except EOFError:
            break
        if line == '':
            continue
        if line.startswith('?'):
            # treat line as a search query
            line = line[1:].strip()
            args = line.split(' ')
            args = [x for x in args if x != '']
            if len(args) == 0:
                continue
            found = str_seen
            # repeatedly narrow to only include results where the string `arg` shows up somewhere in str(p)
            for arg in args:
                found = {str_p:p for str_p,p in found.items() if arg in str_p}
            for p in found.values():
                print(p.pretty())
        else:
            # treat line as a program to execute
            try:
                p = StackProgram.parse(line,search_state)
            except (ProgramFail,ParseError) as e:
                mlb.red(e)
                continue

            seen_str = mlb.mk_green('[seen this stack state]') if p.stack_state in seen else mlb.mk_red('[not seen]')
            print(f'{p.pretty()} {seen_str}')

