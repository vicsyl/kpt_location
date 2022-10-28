import argparse
import re
import subprocess
import traceback
from time import sleep


def run_command(cmd):

  print(f"Running '{cmd}'")
  process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
  return process
  # if process.returncode != 0:
  #   print(f"WARNING: return code: {process.returncode}")
  #   print(f"std_out: {str(process.stdout)}")
  #   print(f"std_err: {str(process.stderr)}")
  #   return None
  # else:
  #   print("OK")
  #   print(f"std_out: {str(process.stdout)}")
  #   print(f"std_err: {str(process.stderr)}")
  #   return process


def run(gpus):
  run_command(f"echo running on {gpus}")


def analyze(process):

  max_iters = 1000000
  counter = 0
  free_counters = [-1] * 16
  cpu_number_re = re.compile(".*\|\s*(\d+).*\.\.\..*On")
  cpu_load_re = re.compile(".*\|\s*(\d+)MiB\s+/\s+(\d+)MiB.*(\d+)%")
  cpu_number = None
  empty_lines_all = 0
  for counter in range(max_iters):
    sleep(0.90)
    print(f"iteration {counter + 1}")
    empty_lines_iter = 0
    while empty_lines_iter < 1000:
      line = process.stdout.readline().strip()
      line = str(line)
      if len(line) > 3:
        print(f"{len(line)}, {line}")
      else:
        empty_lines_all += 1
        if empty_lines_all % 10000 == 0:
          print(f"{empty_lines_all} empty lines")
      m = cpu_number_re.search(line)
      if m and len(m.groups()) > 0 and m.group(1):
        number = int(m.group(1))
        if number < 8:
          cpu_number = number
        else:
          print(f'WARNING: CPU number high: {cpu_number}')
        continue

      m = cpu_load_re.search(line)
      if m and len(m.groups()) > 0 and m.group(1):
        mem_used = int(m.group(1))
        mem_all = int(m.group(2))
        load = int(m.group(3))
        free = mem_all - mem_used > 8000 and load < 10
        if cpu_number is None:
          print("WARNING, no cpu")
        else:
          if free:
            print(f"msg: it's free!! {cpu_number}")
            if free_counters[cpu_number] == -1:
              free_counters[cpu_number] = 1
            else:
              free_counters[cpu_number] += 1
              free_gpus = [i for i, c in enumerate(free_counters) if c > 5]
              if len(free_gpus) > 1:
                print(f"let's run on {free_gpus}")
                return free_gpus

          else:
            free_counters[cpu_number] = 0

  print("finished, no gpus free")
  return None


def do_loop(max_iter):

  for i in range(max_iter):
    try:
      #process = run_command("nvidia-smi -l 1")
      process = run_command("cat ./test_data/nvidia1.txt")
      if process == None:
        print("didn't succeed, will retry")
        continue
      gpus = analyze(process)
      if gpus is None:
        print("no gpus, will retry")
      else:
        print("running ...")
        run(gpus)
    except:
      traceback.print_exc()
    print(f"loop no. finished {i + 1}")

  print("All finished")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Schedule your processes')
  # todo "queue", dir with configs
  parser.add_argument('--max_iter', help='max iter, ideally number of processes', required=True)
  args = parser.parse_args()
  do_loop(int(args.max_iter))
