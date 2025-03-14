#!/Users/xiaokangsun/local_py_proj/Visual-tactile-fusion/.venv/bin/python
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------
#   Author:         E:V:A
#   Date:           2018-02-15
#   Version:        1.0.4
#   License:        GPLv3
#   URL:            https://github.com/E3V3A/pip-date/
#   Description     Show the install date of all pip-installed python3 packages
#----------------------------------------------------------------------
#  ToDo:
#   [ ] better RegEx for "pip"
#   [/] add flake8 QA ignore comments 
#   [/] fix rounding of floats in [a/c/m]Time
#   [ ] Add CLI options:
#       - [ ] '-d'          : Enable extra debug info
#       - [ ] '-e'          : Show "env" column to display 'virtualenv' name
#       - [ ] '-f'          : Force to use opposite (to detected) ctime method for FS
#       - [ ] '-n'          : Disable colors
#       - [ ] '-h'          : THIS help/usage message
#       - [ ] '-v'          : THIS program version
#       - [ ] '-t <days>'   : To highlight packages installed <days> ago
#
# NOTES:
#----------------------------------------------------------------------
#    [a/c/m]time
#    ------------------------------------------------------------------
#    On Windows (via Cygwin & Python3):
#      The creation time is:       aTime           .CreationTime === .LastAccessTime in Poweshell, but known as "access" time in Linux)
#      The modification time is:   mTime == cTime  .LastWriteTime in Poweshell
#
#    On Linux:
#      The creation time is:       cTime
#      The modification time is:   mTime
#      The access time is:         aTime           (normally not used)
#
#    ==> For seeing last modification time, use "cTime" on Windows FS's, and "mTime" on *linux FS's
#----------------------------------------------------------------------
#  References:
#   [1] https://linuxhandbook.com/file-timestamps/
#   [2] https://www.unixtutorial.org/atime-ctime-mtime-in-unix-filesystems/
#----------------------------------------------------------------------
import re, os, sys, platform    # noqa: E401
#import subprocess
import site, pkg_resources      # noqa: E401
from datetime import datetime
from datetime import timedelta
#from time import strftime

__author__ = "E:V:A (E3V3A)"
__copyright__ = "GPLv3 2022"
#__credits__ = ["https://github.com/E3V3A/pip-date/"]
__version__ = '1.0.4'

#----------------------------------------------------------
# OS Check-1
#----------------------------------------------------------
# Apparently for:
# python.exe -c "import os,sys; print('TERM=%s' % os.getenv('TERM'));"
# python -c "import os; print('\n'.join([os.name, os.sys.platform]));"
#   PowerShell/CMD Windows python:  TERM=None
#   PowerShell/CMD Cygwin python:   TERM=cygwin
#   WSL                             TERM=xterm-256color
#----------------------------------------------------------
def is_posix():
    noco = ('dumb', 'xtermm', 'xterm-mono')         # dumb = No VT,  'others' --> $PSStyle.OutputRendering = PlainText
    px_term = os.getenv('TERM')                     # [cygwin, xterm, xterm-color, xterm-256color]
    px_name = os.name                               # [posix, nt, ...]
    px_plat = sys.platform                          # [linux, cygwin, win32]

    if px_term in noco: 
        return False
    if ((px_term == 'None') or (px_term == '')):    # For native Windows "consoles" often return "None", since ="".
        if ( (px_name == 'nt') and (px_plat == 'win32') ):
            return True
        else: 
            return False
    # Assume we have a color term
    return True

#----------------------------------------------------------
# OS Check-2
#----------------------------------------------------------
# We need to test how [a/c/m]time works on the OS
def isWinFS():
    if platform.architecture()[1] == "WindowsPE":
        print("Using cTime for WindowsPE\n")
        return True
    else:
        print("Using mTime for Linux FS\n")
        return False

#----------------------------------------------------------
# Text Coloring
#----------------------------------------------------------
# Usage:  print(yellow("This is yellow"))
def color(text, color_code):
    #if self.nposix:
    if not is_posix():
        return text
    # for brighter colors, use "1;" in front of "color_code"
    bright = ''  # '1;'
    return '\x1b[%s%sm%s\x1b[0m' % (bright, color_code, text)

def red(text):    return color(text, 31)            #                   # noqa
def green(text):  return color(text, 32)            # '1;49;32'         # noqa
def bgreen(text): return color(text, '1;49;32')     # bright green      # noqa
def orange(text): return color(text, '0;49;91')     # 31 - looks bad!   # noqa
def yellow(text): return color(text, 33)            #                   # noqa
def blue(text):   return color(text, '1;49;34')     # bright blue       # noqa
def purple(text): return color(text, 35)            # aka. magenta      # noqa
def cyan(text):   return color(text, '0;49;96')     # 36                # noqa
def white(text):  return color(text, '0;49;97')     # bright white      # noqa

#----------------------------------------------------------
# Print Usage
#----------------------------------------------------------
def usage():
    print(" Usage:  %s\n" % os.path.basename(__file__))
    print(" This will return a detailed sorted list of all your installed packages.")
    print(" The command doesn't take any arguments, and is part of the pip-date")
    print(" package. Other commands includeded in this package are:  pipbyday,")
    print(" pip-describe, pyfileinfo and pyOSinfo.\n")
    print(" Please file any bug reports at:")
    print(" https://github.com/E3V3A/pip-date/\n")
    print(" Version:  %s" % __version__)
    print(" License:  GPLv3\n")
    sys.exit(2)

#----------------------------------------------------------
# Print Warning
#----------------------------------------------------------
def print_warning():
    print('\n')
    print('-'*60)
    print(' WARNING!')
    print(' You are missing out on important color coded information!')
    print(' This is because you are probably using a Windows console')
    print(' that is not fully supporting ANSI color sequencies.')
    print(' For best experience, either run this in Cygwin or WSL,')
    print(' or install WinPty, ConEmu or a PowerShell version >6.1.')
    print(' If you do have a POSIX compatible color terminal, then')
    print(' make sure your TERM environment variable is set.')
    #print(' (Usally to \"xterm\".)')
    print('-'*60)

#----------------------------------------------------------
# Print Color Legend
#----------------------------------------------------------
# See:
#   https://github.com/PowerShell/PowerShell/issues/8409
#   https://en.wikipedia.org/wiki/Code_page_437
#   https://en.wikipedia.org/wiki/Box-drawing_character
#   https://en.wikipedia.org/wiki/Block_Elements
# Let's try:
#   2585,               # Look best but is not part of cp437 and thus font dependent & not widely available
#   2580, 25A0, 2588    # IBM-437
#----------------------------------------------------------
def print_legend():
    #cc = u'\u2585' # Unicode Character for a "5/8th box"           # (U+2585) is not part of IBM-437
    #cc = u'\u2588' # Unicode Character for a "full box"            # (U+2588) is part of IBM-437
    #cc = u'\u25A0' # Unicode Character for a "black square"        # (U+25A0) is part of IBM-437
    cc = u'\u2580'  # Unicode Character for a "Upper half block"    # (U+2580) is part of IBM-437
    print("  {} = ERROR (preventing package processing)".format(red(cc)))
    print("  {} = Using a Bad, Deprecated or Non-Standard installation Path".format(purple(cc)))
    print("  {} = Possibly Multiple installations (differing file times)".format(yellow(cc)))
    print("  {} = Recently Changed / Installed (in last 7 days)".format(cyan(cc)))
    print("  {} = Non-PEM-compliant Version string (PEP-0440) | ~/.local install".format(green(cc)))
    print("  {} = A 'setuptools' dependency package".format(blue(cc)))

#----------------------------------------------------------
# Helper Functions
#----------------------------------------------------------
def safe_name(name):
    # Replace runs of non-alphanumeric characters with a single '-'.
    return re.sub('[^A-Za-z0-9]+', '-', name)

def safe_version(version):
    # Convert an arbitrary string to a standard version string
    version = version.replace(' ', '.')
    return re.sub('[^A-Za-z0-9.]+', '-', version)

def to_filename(name):
    # Replace any '-' characters with '_'.
    return name.replace('-', '_')

def test_loc(loc):
    # Test package location to give us some idea of what type of install it came with.
    #  .local           : are usually local user installs in:  $HOME/.local/lib/pythonX.Y/site-packages/...
    #  site-packages    : are usually system user installs (sudo)
    #  dist-packages    : are usually system package-manager installs (apt)
    #  /PATH/           : are usually developer installs using "pip install ."
    if '.local' in loc:
        ploc = bgreen('usr')        # user (unprivileged local install)
    elif 'dist-packages' in loc:
        ploc = 'apt'                # system (apt package-manger installed)
    elif 'site-packages' in loc:
        ploc = 'sys'                # system (user sudo installed)
    else: 
        #ploc = red(loc)            # show actual path
        ploc = red('dev')           # dev (user development install via "pip install .")
    return ploc

def pre2txt(pre):
    # Distribution "precedence" constants:  (../pkg_resources/__init__.py)
    # EGG_DIST, BINARY_DIST, SOURCE_DIST, CHECKOUT_DIST, DEVELOP_DIST  :  [3,2,1,0,-1]
    # { 'egg': 3, 'bin': 2, 'src': 1, 'chk': 0, 'dev': -1 }
    # However, this seem poorly implemented since most packages show "-1" or 3.
    d = ['chk', 'src', 'bin', 'egg', 'dev']
    return d[pre]

def is_canonical(version):
    # Check PEP-0440 Version string compliance:
    # https://www.python.org/dev/peps/pep-0440/
    canrex = r'^([1-9]\d*!)?(0|[1-9]\d*)(\.(0|[1-9]\d*))*((a|b|rc)(0|[1-9]\d*))?(\.post(0|[1-9]\d*))?(\.dev(0|[1-9]\d*))?$'
    return re.match(canrex, version) is not None

def pkgcol(pkgarr):
    #----------------------------------------------------------
    # The <package_name> color require special treatment, because
    # of sorting on key position and getting ljust space.
    # https://packaging.python.org/key_projects/
    cygset  = ['setuptools', 'appdirs', 'packaging', 'pyparsing', 'six']    # Cygwin python3-setuptools dependency packages
    #cygset += ['wheel', 'virtualenv', 'pipenv', 'pip']                      # ...some additional essentials
    #cygset += ['scikit-build', 'distlib']                                   # ...some additional essentials
    # NOTE!
    #   We can't use "pip" because the we're only checking if the string is present in line,
    #   Thus anything with "pip" in it would be caught, so we need a smarter RE here.
    #----------------------------------------------------------
    line = ''
    for i in range(len(pkgarr)):
        line = pkgarr[i]
        for pname in cygset:
            # ToDo: only replace if in 1st word
            if pname in line:
                #pkgarr[i] = re.sub(r'^[a-zA-Z0-9_\-]+', blue(pname), line, 1)
                rx = line.find(pname)
                if rx < 20 and rx != '-1':
                    pkgarr[i] = line.replace(pname, blue(pname), 1)
                    break
    return pkgarr

#----------------------------------------------------------
# MAIN
#----------------------------------------------------------
def main_func():

    print()
    debug    = 0
    #nposix   = 0
    pcnt     = 0
    pkg      = []

    #---------------------------------------
    # CLI arguments
    #---------------------------------------
    narg = len(sys.argv) - 1
    if narg >= 1:
        #pkg = sys.argv[1]
        usage()
    #if is_posix():
    #    nposix = 1
    #---------------------------------------

    # Check [a/c,m]time availability
    useWinStat = False      # Linux ELF based FS system
    if isWinFS():
        useWinStat = True  # WindowsPE FS system

    #---------------------------------------
    #  MacOS: ??
    #  Cygwin: native python, we only have 1 location:
    #       /usr/lib/python3.6/site-packages
    #  Linux Mint (19.1): we have several (3)
    #       /usr/local/lib/python3.6/dist-packages
    #       /usr/lib/python3/dist-packages
    #       /usr/lib/python3.6/dist-packages
    # Also add the unprivileged user's local package location:
    #       $HOME/.local/lib/python3.6/site-packages/
    #---------------------------------------
    try:
        site_loc = site.getsitepackages()                   # [...]
        if debug: print("site_locs (site): ", site_loc)
        site_loc += [site.getusersitepackages()]            # add $HOME/.local/lib/python3.6/site-packages/
        if debug: print("site_locs (all): ", site_loc)
    except AttributeError:
        site_loc = site.USER_SITE

    for d in pkg_resources.working_set:

        try:
            pkg_name = d.project_name                                                               #
            pkg_ver  = d.version                                                                    #
            #pkg_loc  = d.location                                                                  # NOT always a file!
            pkg_typ  = 'n/a' # "wheel" if d.location.is_wheel else "sdist"                          # 'Type'

            if debug:
                pkg_pre  = d.precedence                                                             # 'Prec' [-1..3]
            else:
                pkg_pre  = pre2txt(d.precedence) if (d.precedence != -1) else ''                    # 'Prec' string

            pkg_ins  = d.get_metadata('INSTALLER').strip() if d.has_metadata('INSTALLER') else ''   # get_metadata_lines() '???'
            pkg_whl  = d.get_metadata('WHEEL').strip() if d.has_metadata('WHEEL') else ''           # get_metadata_lines() '???'
            # ^^^^^ This often have multiple lines, we need to format:
            if pkg_whl:
                pw = pkg_whl.split('\n')
                pw = '\n    ' + '\n    '.join(pw,)
                pkg_whl = pw

        except ValueError as e:
            print(red("ERROR:") + " %s" % e)

        #---------------------------------------
        # Get the correct package location
        #---------------------------------------
        # Because d.location doesn't return a file, but only a directory,
        # for certain packages, we also check the "module directory" ???
        try:
            mod_dir = next(d._get_metadata('top_level.txt'))    # module_dir
            pkg_loc = os.path.join(d.location, mod_dir)         #
            os.stat(pkg_loc)                                    #

        except (StopIteration, OSError):
            try:
                pkg_loc = os.path.join(d.location, d.key)
                os.stat(pkg_loc)
            except OSError:
                pkg_loc = d.location
        #---------------------------------------

        if debug:
            print('-'*40)
            print("pkg_loc:  %s: %s  (%s)" % (pkg_name.ljust(20,' '), pkg_loc, pkg_pre))    # pkg_pre always empty ??
            print("pkg_typ:  %s" % pkg_ins)
            print("pkg_whl:  %s" % pkg_whl)
            print("pkg_ins:  %s" % pkg_ins)

        # A work-around for packages with deprecated location(s):
        if ".egg" in pkg_loc: 
            print(purple("Found Bad Path Location for:") + "  %s" % white(pkg_name))
            print("Package Location found at:    %s" % (pkg_loc))

        #---------------------------------------
        # Getting OS Dependent TimeStamps
        #---------------------------------------
        # NOTE:
        # (1) [acm]time as used in variable names HERE, is true for LinuxFS,
        #     but swapped for WindowsFS's
        # (2) In a WindowsPE based FS:
        #       (a) The true "creation" time is the (python stat) "atime"
        #       (b) The true "modification" time is the (python stat) "mtime"=="ctime"
        # (3) In a ELF based LinuxFS:
        #       aTime = access time ...     - Rarely used because of FS performance
        #       mtime = modification time   -
        #       ctime = creation time       - is the "real" last modification time on windows
        #---------------------------------------
        if debug: print("pkg_loc:  %s" % pkg_loc)

        if os.path.exists(pkg_loc):
            if useWinStat:
                tsc = os.path.getctime(pkg_loc)     # WindowsFS:    ctime:  "creation time"
                tsm = os.path.getmtime(pkg_loc)     # WindowsFS:    mtime:  "last modified"
                #tsm = os.path.getatime(pkg_loc)    # WindowsFS:    atime:  "last accessed"
            else:
                tsc = os.path.getctime(pkg_loc)     # LinuxFS (ctime)
                tsm = os.path.getmtime(pkg_loc)     # LinuxFS (mtime)
        else:
            #print(red("Skipping Bad Path of:") + " %s: \t%s" % (pkg_name, pkg_path))   # to_filename(pkg_name))
            print(red("Skipping Bad Path of:") + " %s: \t%s" % (pkg_name, pkg_loc))     # to_filename(pkg_name))
            continue

        #------------------------------------------------------------------------------------
        # Processing Time Stamps
        #------------------------------------------------------------------------------------
        # Logic:  
        #   1. IF  ( mTime > (K * cTime) )   THEN  highlight mTime  ELSE  don't show  # (1)
        #   2. IF  ( cTime < '1-week-ago' )  THEN  highlight cTime                    # (2) 
        #
        #------------------------------------------------------------------------------------
        pkg_ctime = datetime.fromtimestamp(tsc).strftime("%Y-%m-%d  %H:%M:%S").strip()  # str
        pkg_mtime = datetime.fromtimestamp(tsm).strftime("%Y-%m-%d  %H:%M:%S").strip()  # str

        # NOTE:  If TS differs by < 1 second, it will not show. To do so, test:  tsc == tsm
        max_tdelta = 60                         # Max allowed time difference: ~60s
        tdelta = abs(tsc - tsm)                 # Calculate time difference
        #if pkg_ctime != pkg_mtime:             # This is too restrictive (OS need seconds to install)
        if tdelta > max_tdelta:                 #
            pkg_mtime = yellow(pkg_mtime)       # (1) Highlight packages with different creation vs modification Times
        else:
            pkg_mtime = ''                      # Skip those where: mTime ~= cTime

        ctNow = datetime.now()                  # Time: "now"
        ct7dy = ctNow - timedelta(days=7)       # Time: 1-week-ago
        pctim = datetime.fromtimestamp(tsc)     # Time: file cTime (Time Stamp)
        if pctim > ct7dy:
            pkg_ctime = cyan(pkg_ctime)         # (2) Highlight packages recently created (cTime < 1-week-ago)
        #------------------------------------------------------------------------------------

        if pkg_pre:
            pkg_pre = yellow(pkg_pre) + " "     # [egg,...]

        if not is_canonical(pkg_ver):
            pkg_ver = green(pkg_ver) + " "*3    # green + ugly color-code-length hack when using colors...

        # ugly test here... must be a better way
        if 'bdist_wheel' in pkg_whl:
            pkg_typ = 'wheel'
        else:
            #pkg_typ = white('sdist')
            pkg_typ = orange('sdist')

        pkg_loc = test_loc(pkg_loc)

        pcnt += 1
        pkg  += ["{:20}   {:<20}   {:<20}   {:<16}   {:6}   {:<4}   {:5}   {:3}".format(pkg_name.ljust(20,' '), pkg_ctime, pkg_mtime, pkg_ver, pkg_ins, pkg_pre, pkg_typ, pkg_loc)] # noqa

    if useWinStat:
    #   header_str = "{:20}   {:20}   {:20}   {:16}   {:6}   {:4}   {:5}   {:3}".format('Package'.ljust(20, ' '), 'LastModified (mTime)', 'FirstSeen (aTime)', 'Version', 'Inst', 'Prec', 'Type ', 'Loc')
        header_str = "{:20}   {:20}   {:20}   {:16}   {:6}   {:4}   {:5}   {:3}".format('Package'.ljust(20, ' '), 'Installed (cTime)', 'LastModified (mTime)', 'Version', 'Inst', 'Prec', 'Type ', 'Loc')
    else:
        header_str = "{:20}   {:20}   {:20}   {:16}   {:6}   {:4}   {:5}   {:3}".format('Package'.ljust(20, ' '), 'LastModified (mTime)', 'FirstSeen (cTime)', 'Version', 'Inst', 'Prec', 'Type ', 'Loc')

    hlen = len(header_str)
    print('\n' + header_str)

    print("-"*hlen)
    spkg = sorted(pkg, key=str.lower)
    spkg = pkgcol(spkg)
    print('\n'.join(spkg))

    print("-"*hlen)
    print_legend()

    print("-"*hlen)
    print("Found %d packages." % pcnt)

    #if not nposix:
    if not is_posix():
        print_warning()

    print("\nDone!")


if __name__ == "__main__":
    main_func()
    sys.exit(0)
