#!/bin/bash

textifyDuration() {
   local duration=$1
   local shift=$duration
   local secs=$((shift % 60));  shift=$((shift / 60));
   local mins=$((shift % 60));  shift=$((shift / 60));
   local hours=$shift
   local splur; if [ $secs  -eq 1 ]; then splur=''; else splur='s'; fi
   local mplur; if [ $mins  -eq 1 ]; then mplur=''; else mplur='s'; fi
   local hplur; if [ $hours -eq 1 ]; then hplur=''; else hplur='s'; fi
   if [[ $hours -gt 0 ]]; then
      txt="$hours hour$hplur, $mins minute$mplur, $secs second$splur"
   elif [[ $mins -gt 0 ]]; then
      txt="$mins minute$mplur, $secs second$splur"
   else
      txt="$secs second$splur"
   fi
   echo "$txt (from $duration seconds)"
}


#
# Set Colors
#

purple=$(tput setaf 171)
red=$(tput setaf 1)
green=$(tput setaf 76)
tan=$(tput setaf 3)

#
# Headers and  Logging
#

e_header() { printf "\n${bold}${purple}==========  %s  ==========${reset}\n" "$@"
}
e_arrow() { printf "➜ $@\n"
}
e_success() { printf "${green}✔ %s${reset}\n" "$@"
}
e_error() { printf "${red}✖ %s${reset}\n" "$@"
}
e_warning() { printf "${tan}➜ %s${reset}\n" "$@"
}
e_underline() { printf "${underline}${bold}%s${reset}\n" "$@"
}
e_bold() { printf "${bold}%s${reset}\n" "$@"
}
underline=$(tput sgr 0 1)
bold=$(tput bold)
blue=$(tput setaf 38)
reset=$(tput sgr0)
e_note() { printf "${underline}${bold}${blue}Note:${reset}  ${blue}%s${reset}\n" "$@"
}
