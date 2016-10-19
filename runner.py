from subprocess import call
import time
import argparse


if __name__ == '__main__':

    call(['sudo', 'echo'])

    parser = argparse.ArgumentParser(description="Test Runner")
    parser.add_argument('type', nargs='+', help='Must be either tx or rx')

    args = parser.parse_args()

    run_type = vars(args)['type'][1]

    print "got type", run_type

    assert run_type == 'tx' or run_type == 'rx'

    if run_type == 'rx':
        runnerfo = open('_rx_targets', 'r')
    if run_type == 'tx':
        runnerfo = open('_tx_targets', 'r')

    targtext = runnerfo.read(999999)
    runnerfo.close()

    commands = targtext.split(' ')

    for i in range(len(commands)):
        commands[i] = './' + commands[i]


    target_time = 65

    runs = []
    for com in commands:
        t1 = time.time()
        print "starting at", t1

        call(['sudo', com])
        t2 = time.time()
        print "ending at", t2

        delta = t2-t1

        if delta >= target_time:
            print "ZOMG this run took TOO LONG", delta, "instead of", target_time
        else:
            time.sleep(target_time-delta)

        runs.append(t2-t1)



    print "all runs"
    print repr(runs)
