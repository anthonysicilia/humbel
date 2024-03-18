import pickle

write = True

if __name__ == '__main__':
    data = pickle.load(open('default-slp-qa.pkl', 'rb'))
    test = None
    i = 1
    for (ptype, ttype, prompt, resp) in data:
        if test != ttype:
            test = ttype
            i = 1
        if write:
            out = open('show-default-slp-qa.txt', 'a')
            print = lambda *args: out.write(' '.join(args) + '\n')
        print('=' * 10, f'New Example {ptype} {ttype} {i}', '=' * 10, '(hit enter to see next example)')
        print('Prompt:', prompt)
        print('Response:', resp)
        if not write:
            input()
        i += 1
