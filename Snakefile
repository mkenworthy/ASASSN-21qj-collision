rule proc_atlas:
     input:
        "src/data/atlas/job211831.txt"
     output:
        "src/data/obs_ATLAS.ecsv"
     conda:
        "environment.yml"
     script:
        "src/scripts/convert_atlas.py"

rule proc_asassn:
     input:
        "src/data/asassn/light_curve_410e0d3c-687e-40a3-b7cb-af0057695e0b.csv"
     output:
        "src/data/obs_ASASSN.ecsv"
     conda:
        "environment.yml"
     script:
        "src/scripts/convert_asassn.py"

rule proc_neowise:
     input:
        "src/data/neowise/ASASSN-21qj_2013-2022.tbl"
     output:
        "src/data/obs_NEOWISE.ecsv"
     conda:
        "environment.yml"
     script:
        "src/scripts/convert_neowise.py"

rule proc_aavso:
     input:
        "src/data/aavso/aavsodata_63e2220f49f39.txt"
     output:
        "src/data/obs_AAVSO.ecsv"
     conda:
        "environment.yml"
     script:
        "src/scripts/convert_aavso.py"

rule proc_wise_col_temp:
     input:
        "src/data/obs_NEOWISE.ecsv"
     output:
        "src/data/NEOWISE_coltemp.ecsv"
     conda:
        "environment.yml"
     script:
        "src/scripts/neowise_coltemp.py"

rule proc_gyro_age_posterior:
     input:
        "src/scripts/asassn21qj.py"
     output:
        "src/data/gyro_age_posterior.pkl"
     conda:
        "environment_gyro.yml"
     script:
        "src/scripts/calc_gyro_age_posterior.py"


rule calc_gyro_age:
     input:
        "src/data/gyro_age_posterior.pkl"
     output:
        "src/tex/output/gyro_age.txt"
     conda:
        "environment_gyro.yml"
     script:
        "src/scripts/calc_gyro_age.py"
