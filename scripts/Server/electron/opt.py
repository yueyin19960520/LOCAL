import os
import datetime
import argparse
import random
import time


def construct_run_list(root):
    l1 = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root,f))]
    fi = []
    un = []
    for i in l1:
        kpoints = os.path.join(root, i, "KPOINTS")
        outcar = os.path.join(root, i, "OUTCAR")
        if os.path.exists(kpoints): 
            content = os.popen('grep "reached required accuracy - stopping structural energy minimisation" %s' % (outcar)).read()
            if 'reached required accuracy - stopping structural energy minimisation' in content:
                fi.append(i)
            #else:
                #un.append(i)
        else:
            un.append(i)

    random.shuffle(un)
    return un


def run(p_dir, sub_dir, root): #root = opt_pool
    # Three stages
    opt = os.path.join(root, sub_dir, "opt")
    static = os.path.join(root, sub_dir, "static")
    cohp = os.path.join(root, sub_dir , "cohp")
    for temp_path in [opt, static, cohp]:
        os.mkdir(temp_path)

    # Three keywords
    opt_keyword = "reached required accuracy - stopping structural energy minimisation"
    static_keyword = "aborting loop because EDIFF is reached"
    finished_keyword = "finished in"

    # INPUT files
    INPUT_incar_opt = os.path.join(p_dir,"INPUT","INCAR.opt")
    INPUT_kpoints = os.path.join(p_dir,"INPUT","KPOINTS")    
    INPUT_incar_static = os.path.join(p_dir, "INPUT", "INCAR.static")
    INPUT_lobsterin = os.path.join(p_dir, "INPUT", "lobsterin")
    
    # OPT files
    opt_incar = os.path.join(opt,"INCAR")
    opt_kpoints = os.path.join(opt,"KPOINTS")
    opt_outcar = os.path.join(opt, "OUTCAR")
    opt_contcar = os.path.join(opt, "CONTCAR")

    # Static files
    static_incar = os.path.join(static,"INCAR")
    static_kpoints = os.path.join(static,"KPOINTS")
    static_outcar = os.path.join(static, "OUTCAR")
    static_contcar = os.path.join(static, "CONTCAR")
    static_wavecar = os.path.join(static, "WAVECAR")
    static_vasprun = os.path.join(static, "vasprun.xml")
    static_contcar = os.path.join(static, "CONTCAR")
    static_potcar = os.path.join(static, "POTCAR")

    # COHP files
    cohp_lobsterout = os.path.join(cohp, "lobsterout")
    cohp_contcar = os.path.join(cohp, "CONTCAR")
    cohp_outcar = os.path.join(cohp, "OUTCAR.opt")
    cohp_icohplist = os.path.join(cohp, "ICOHPLIST.lobster")
    cohp_lobsterin = os.path.join(cohp,"lobsterin")


    ###########################
    ######### Opt Part ########
    ###########################
    # Get into the current sub_dir
    os.chdir(opt)

    # Check if this sub_dir is running or finished
    if os.path.isfile(opt_kpoints): 
        return

    # Check if Incar exist, rerun may need new INCAR
    # Copy INCAR from INPUT
    if not os.path.isfile(opt_incar):
        os.system("cp %s %s"%(INPUT_incar_opt, opt_incar)) 

    # Copy KPOINTS from INPUT
    os.system("cp %s %s"%(INPUT_kpoints, opt_kpoints))

    # Make up POTCAR by vaspkit command
    os.system('echo -e "1\n103\n" | vaspkit >/dev/null 2>&1') 
    
    # VASP run command with corresponding module load
    os.system("module load VASP/6.3.0 && mpirun vasp_std > vasp.out 2>vasp.err")

    ###########################
    ####### Static Part #######
    ###########################
    # Get the content of the OUTCAR
    if os.path.exists(opt_outcar):
        with open(opt_outcar, "r") as f:
            opt_content = f.read()
    else:
        opt_content = []
    
    # If the structure optimization is finished, the static could start!
    if opt_keyword in opt_content and os.path.exists(opt_contcar) and os.path.getsize(opt_contcar)>0:
        # current static directory and create in the current sub_dir(opt)
        
        os.mkdir(static)

        # new_OUTCAR for avoiding the OUTCAR from static calculation
        # POSCAR which is the CONTCAR in the opt calculation
        temp_outcar = os.path.join(static, "OUTCAR.opt")
        static_poscar = os.path.join(static, "POSCAR")
        os.system("cp %s %s"%(opt_outcar, temp_outcar))
        os.system("cp %s %s"%(opt_contcar, static_poscar))

        # romove others keeping the static folder
        os.system("rm !(static)")

        # Get into the static directory
        os.chdir(static)



        # Copy INCAR from INPUT
        os.system("cp %s %s"%(INPUT_incar, static_incar))

        # Copy KPOINTS from INPUT
        os.system("cp %s %s"%(INPUT_kpoints, static_kpoints))

        # Make up POTCAR by vaspkit command
        os.system('echo -e "1\n103\n" | vaspkit >/dev/null 2>&1')

        # VASP run command with corresponding module load
        os.system("module load VASP/6.3.0 && mpirun vasp_std > vasp.out 2>vasp.err")

    ###########################
    ######### COHP Part #######
    ###########################

    # Get the content of the OUTCAR
    if os.path.exists(static_outcar):
        with open(static_outcar, "r") as f:
            static_content = f.read()
    else:
        static_content = []

    # If the static calculaton is finished, the cohp could start!
    if static_keyword in static_content and os.path.exists(static_contcar) and os.path.getsize(static_contcar)>0:
        # current static directory and create in the current sub_dir(opt)
        
        os.mkdir(cohp)

        # Prepareing the COHP calculation needed files
        os.system("cp %s %s"%(static_outcar, cohp))
        os.system("cp %s %s"%(static_contcar, cohp))
        os.system("cp %s %s"%(opt_outcar, cohp))
        os.system("cp %s %s"%(static_kpoints, cohp))
        os.system("cp %s %s"%(static_wavecar, cohp))
        os.system("cp %s %s"%(static_potcar, cohp))
        os.system("cp %s %s"%(static_vasprun, cohp))
        
        # remove everthing from the static folder
        os.system("rm *")

        # Get into the COHP
        os.chdir(cohp)



        if os.path.isfile(os.path.join(cohp, "lobsterin")):
            return
        
        # COPY the lobsterin for INPUT
        os.system("cp %s %s"%(INPUT_lobsterin, cohp_lobsterin))

        # Run the Lobster for COHP Calculation
        os.system("lobster") 




    ###########################
    ####### Finish Part #######
    ###########################

        

        if os.path.exists(finished_lobsterout):
            with open(finished_lobsterout, "r") as f:
                finished_content = f.read()
        else:
            finished_content = ""
        
        if finished_keyword in finished_content and os.path.exists(finished_icohplist) and os.path.getsize(finished_icohplist)>0:
            finished = opt
            os.system("cp %s %s"%(finished_outcar, finished))
            os.system("cp %s %s"%(finished_contcar, finished))
            os.system("cp %s %s"%(finished_icohplist, finished))



if __name__ == "__main__":
    p_dir=os.getcwd()
    root = os.path.join(p_dir,"opt_pool")  #name is opt_pool????
    
    #sleep_t = random.randint(10,30)
    #time.sleep(sleep_t*5)
    
    os.system("module load VASP/6.3.0")
   
    while True:
        rerun_list = construct_run_list(root)
        if rerun_list == []:
            break
        else:
            run(p_dir, rerun_list[0], root)
