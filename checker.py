import argparse
import os
import re
import subprocess
import traceback
from time import sleep

def run_command_list(cmd_list):

  print(f"Running '{cmd_list}'")
  process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE)
  return process


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

  def get_name():
    process = run_command("./get_name.sh")
    sleep(0.01)
    name = process.stdout.readline().strip()
    return str(name)[2:-1]

  def get_file():
    files = [f"queue/{f}" for f in os.listdir("./queue") if os.path.isfile(f"queue/{f}") and f.endswith(".yaml")]
    if len(files) == 0:
      return None
    else:
      files.sort(key=os.path.getmtime)
      return files[-1]

  def get_config_path(name, file_path):
    full_path = f"./work/{name}"
    run_command(f"mkdir {full_path}")
    run_command(f"mv {file_path} {full_path}")
    return f"{full_path}/{file[6:]}"

  print(f"echo running on {gpus}")

  file = get_file()
  if file is None:
    print("no file found")
    return None
  name = get_name()
  config_path = get_config_path(name, file)
  devices = " ".join([str(g) for g in gpus])

  final_cmd = ["./sch_train.sh", name, config_path, devices]
  #final_cmd = ["python", "-u", "./train.py", "--config", config_path, "--name", name, "--devices", devices]
  print(f"running {final_cmd}")
  return run_command_list(final_cmd)


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
        empty_lines_iter += 1
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
      process = run_command("nvidia-smi -l 1")
      # process = run_command("cat ./test_data/nvidia1.txt")
      if process == None:
        print("didn't succeed, will retry")
        continue
      gpus = analyze(process)
      if gpus is None:
        print("no gpus, will retry")
      else:
        print("running ...")
        r = run(gpus)
        if r is None:
          print("probably no file found, quitting")
          return
        secs = 300
        print(f"will sleep for {secs} seconds ...")
        sleep(secs)
    except KeyboardInterrupt:
      traceback.print_exc()
      print("quitting because of KeyInterrupt")
    except:
      traceback.print_exc()
    print(f"loop no. finished {i + 1}")

  print("All finished")


def main():

  parser = argparse.ArgumentParser(description='Schedule your processes')
  # todo "queue", dir with configs
  parser.add_argument('--max_iter', help='max iter, ideally number of processes', required=True)
  args = parser.parse_args()
  do_loop(int(args.max_iter))


if __name__ == "__main__":
  main()
