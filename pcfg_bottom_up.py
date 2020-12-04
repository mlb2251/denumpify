from compiletime import pre_synth
import numpy as np

def pcfg_bottom_up(fns,cfg):
    priors,consts,vars,fns = pre_synth(fns,cfg)


    priors = {k:int(v) for k,v in priors.items()}

    step = 1
    target = 0

    terminals = []
    nonterminals = []


    while True:
        print(f"{target:=}")
        terminals.sort(key=lambda x:-x.ll)
        frozen_terminals = terminals[:] # this shallow copy by "[:]" is v imp
        for nonterminal in nonterminals:
            if nonterminal.prior <= target:
                continue

            args_cands = get_arg_candidates(
                min_ll = target - nonterminal.prior, # minimum increases a little
                arginfos_remaining = nonterminal.arginfos,
                args_so_far = [],
                terminals = frozen_terminals,
                )
            for args_cand in args_cands:
                # args_cand :: [Terminal]
                assert nonterminal.prior + sum([term.ll for term in args_cand]) >= target
                expr = Expr.from_nonterminal(nonterminal, args_cand)
                res = expr.evaluate(env)
                # TODO do some observational equiv checking here (careful to assert relative lls so you dont throw out an optimal choice)
                terminals.append(Terminal.from_nonterminal(nonterminal,args_cand))



        target -= 1


def get_arg_candidates(
    min_ll, # lowest allowed ll, dont return candidates below this
    arginfos_remaining, # list of ArgInfo objects
    args_so_far, # list of Terminal objects
    terminals, # list of Terminal objects
    ):
    if len(arginfos_remaining) == 0:
        yield args_so_far # this is the base case that actually returns a real value!
    arginfo = arginfos_remaining[0]
    for terminal in terminals:
        if terminal.ll < min_ll:
            return # generator stops. Note we assume terminals are sorted by ll
        if not arginfo.allow_terminal(terminal):
            continue # skip terminals that dont fit any specified requirements of this argument
        # TODO itd be easy to add an extra ll penalty here much like allow_terminal but a cost instead of outright rejection
        yield from get_arg_candidates(
            min_ll = min_ll-terminal.ll, # minimum increases a little
            arginfos_remaining = arginfos_remaining[1:],
            args_so_far = args_so_far + [terminal],
            terminals = terminals,
            )




class Terminal:
    def __init__(self,expr,val):
        self.expr = expr
        self.val = val
        self.ll = expr.nonterminal.prior + sum([term.ll for term in args])

    @staticmethod
    def from_nonterminal(nonterminal, args):
        # args :: [Terminal]
        assert len(args) == nonterminal.argc
        expr = Expr(nonterminal,[term.expr for term in args])
        return Terminal(expr, ll)

class Nonterminal:
    arginfos # list of ArgInfo
    argc
    prior

class ArgInfo:
    allow_terminal = lambda x:True

class Expr:
    def __init__(self,nonterminal, args, ll):
        self.nonterminal = nonterminal
        self.args = args # [Expr]
        self.ll = ll
    @staticmethod
    def from_nonterminal(nonterminal, args):
        # args :: [Terminal]
        assert len(args) == nonterminal.argc
        return Expr(nonterminal,[term.expr for term in args], nonterminal.prior + [term.ll for term in args])



def foo():
    raise NotImplementedError


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
    raise NotImplementedError
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