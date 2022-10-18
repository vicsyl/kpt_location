import subprocess
from urls import urls_to_download
from dataset_utils import clean_scene


def download_zips(zips_to_download_start=0, zips_to_download_end=10):

  for url in urls_to_download[zips_to_download_start:zips_to_download_end]:
    zip_file = url[url.rfind("/") + 1:]
    print(zip_file)
    run_command(f"curl {url} --output ./zips/{zip_file}")
    print("unzipping ...")
    run_command(f"unzip -q ./zips/{zip_file}")
    #print("removing: {}".format(zip_file))
    run_command(f"mv {zip_file[:-4]} ./unzips/")
    scene_dir = f"./unzips/{zip_file[:-4]}"
    clean_scene(scene_dir)


def run_command(cmd):
  print(f"Running '{cmd}'")
  #bashCommand = "ls -al ."
  process = subprocess.run(cmd.split(), stdout=subprocess.PIPE)
  if process.returncode != 0:
    print(f"WARNING: return code: {process.returncode}")
    print(f"std_out: {str(process.stdout)}")
    print(f"std_err: {str(process.stderr)}")
  else:
    print("OK")
    print(f"std_out: {str(process.stdout)}")
    print(f"std_err: {str(process.stderr)}")


if __name__ == "__main__":
  download_zips(0, 2)
