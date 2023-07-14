#!/usr/bin/env bash

# reference:
# https://gpt-index.readthedocs.io/en/latest/end_to_end_tutorials/chatbots/building_a_chatbot.html

# sh run.sh --stage -1 --stop_stage -1 --system_version windows
# sh run.sh --stage 2 --stop_stage 2 --system_version windows

system_version=windows
verbose=true;
stage=-1
stop_stage=3


# parse options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
    --*) name=$(echo "$1" | sed s/^--// | sed s/-/_/g);
      eval '[ -z "${'"$name"'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;
      old_value="(eval echo \\$$name)";
      if [ "${old_value}" == "true" ] || [ "${old_value}" == "false" ]; then
        was_bool=true;
      else
        was_bool=false;
      fi

      # Set the variable to the right value-- the escaped quotes make it work if
      # the option had spaces, like --cmd "queue.pl -sync y"
      eval "${name}=\"$2\"";

      # Check that Boolean-valued arguments are really Boolean.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;

    *) break;
  esac
done

work_dir="$(pwd)"

data_dir="${work_dir}/data_dir"

mkdir -p "${data_dir}"

export PYTHONPATH="${work_dir}/../../.."

if [ $system_version == "windows" ]; then
  #source /data/local/bin/OpenLangChain/bin/activate
  alias python3='C:/Users/tianx/PycharmProjects/virtualenv/OpenLangChain/Scripts/python.exe'
elif [ $system_version == "centos" ] || [ $system_version == "ubuntu" ]; then
  alias python3='/data/local/bin/OpenLangChain/bin/python3'
fi


if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  $verbose && echo "stage -1: download data"
  cd "${data_dir}" || exit 1;

  wget "https://www.dropbox.com/s/948jr9cfs7fgj99/UBER.zip?dl=1" -O UBER.zip
  unzip UBER.zip
  rm -rf UBER.zip
  rm -rf __MACOSX

fi


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  $verbose && echo "stage 0: build storage"
  cd "${work_dir}" || exit 1;

  python3 1.build_storage.py
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  $verbose && echo "stage 1: build chatbot agent"
  cd "${work_dir}" || exit 1;

  python3 2.build_chatbot_agent.py
fi
