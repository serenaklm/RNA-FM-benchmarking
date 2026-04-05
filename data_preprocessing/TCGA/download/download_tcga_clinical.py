#!/usr/bin/env python3
"""
Download all TCGA clinical data (including treatments) for all TCGA projects
in parallel and save each to a CSV file.
"""

import os
import requests
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# --- Configuration ---
OUTPUT_DIR = "/data/rbg/users/ashyamal/data/expanded_TCGA_clinical_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_WORKERS = 8

GDC_API = "https://api.gdc.cancer.gov/cases"

TCGA_PROJECTS = [
    'TCGA-BRCA', 'TCGA-GBM','TCGA-OV','TCGA-LUAD', 'TCGA-UCEC', 'TCGA-KIRC',
    'TCGA-HNSC','TCGA-LGG','TCGA-THCA','TCGA-LUSC','TCGA-PRAD','TCGA-SKCM',
    'TCGA-COAD','TCGA-STAD','TCGA-BLCA','TCGA-LIHC','TCGA-CESC','TCGA-KIRP',
    'TCGA-SARC','TCGA-LAML','TCGA-ESCA','TCGA-PAAD','TCGA-PCPG','TCGA-READ',
    'TCGA-TGCT','TCGA-THYM','TCGA-KICH','TCGA-ACC','TCGA-MESO','TCGA-UVM',
    'TCGA-DLBC','TCGA-UCS','TCGA-CHOL'
]

FIELDS = [
    "case_id",
    "submitter_id",
    "project.project_id",
    #"days_to_lost_to_followup",
    
    # Demographic fields
    "demographic.age_at_index",
    "demographic.age_is_obfuscated",
    "demographic.cause_of_death",
    "demographic.cause_of_death_source",
    "demographic.country_of_birth",
    "demographic.country_of_residence_at_enrollment",
    "demographic.days_to_birth",
    "demographic.days_to_death",
    "demographic.demographic_id",
    "demographic.education_level",
    "demographic.ethnicity",
    "demographic.gender",
    "demographic.marital_status",
    "demographic.occupation_duration_years",
    "demographic.population_group",
    "demographic.premature_at_birth",
    "demographic.race",
    "demographic.submitter_id",
    "demographic.vital_status",
    "demographic.weeks_gestation_at_birth",
    "demographic.year_of_birth",
    "demographic.year_of_death",
    
    # Diagnosis fields
    "diagnoses.adrenal_hormone",
    "diagnoses.age_at_diagnosis",
    "diagnoses.ajcc_clinical_m",
    "diagnoses.ajcc_clinical_n",
    "diagnoses.ajcc_clinical_stage",
    "diagnoses.ajcc_clinical_t",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_serum_tumor_markers",
    "diagnoses.ajcc_staging_system_edition",
    "diagnoses.ann_arbor_b_symptoms",
    "diagnoses.ann_arbor_b_symptoms_described",
    "diagnoses.ann_arbor_clinical_stage",
    "diagnoses.ann_arbor_extranodal_involvement",
    "diagnoses.ann_arbor_pathologic_stage",
    "diagnoses.best_overall_response",
    "diagnoses.burkitt_lymphoma_clinical_variant",
    "diagnoses.calgb_risk_group",
    "diagnoses.cancer_detection_method",
    "diagnoses.child_pugh_classification",
    "diagnoses.clark_level",
    "diagnoses.classification_of_tumor",
    "diagnoses.cog_liver_stage",
    "diagnoses.cog_neuroblastoma_risk_group",
    "diagnoses.cog_renal_stage",
    "diagnoses.cog_rhabdomyosarcoma_risk_group",
    "diagnoses.contiguous_organ_invaded",
    "diagnoses.days_to_best_overall_response",
    "diagnoses.days_to_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_last_known_disease_status",
    "diagnoses.days_to_recurrence",
    "diagnoses.diagnosis_id",
    "diagnoses.diagnosis_is_primary_disease",
    "diagnoses.double_expressor_lymphoma",
    "diagnoses.double_hit_lymphoma",
    "diagnoses.eln_risk_classification",
    "diagnoses.enneking_msts_grade",
    "diagnoses.enneking_msts_metastasis",
    "diagnoses.enneking_msts_stage",
    "diagnoses.enneking_msts_tumor_site",
    "diagnoses.ensat_clinical_m",
    "diagnoses.ensat_pathologic_n",
    "diagnoses.ensat_pathologic_stage",
    "diagnoses.ensat_pathologic_t",
    "diagnoses.esophageal_columnar_dysplasia_degree",
    "diagnoses.esophageal_columnar_metaplasia_present",
    "diagnoses.fab_morphology_code",
    "diagnoses.figo_stage",
    "diagnoses.figo_staging_edition_year",
    "diagnoses.first_symptom_longest_duration",
    "diagnoses.first_symptom_prior_to_diagnosis",
    "diagnoses.gastric_esophageal_junction_involvement",
    "diagnoses.gleason_grade_group",
    "diagnoses.gleason_grade_tertiary",
    "diagnoses.gleason_patterns_percent",
    "diagnoses.gleason_score",
    "diagnoses.goblet_cells_columnar_mucosa_present",
    "diagnoses.icd_10_code",
    "diagnoses.igcccg_stage",
    "diagnoses.inpc_grade",
    "diagnoses.inpc_histologic_group",
    "diagnoses.inrg_stage",
    "diagnoses.inss_stage",
    "diagnoses.international_prognostic_index",
    "diagnoses.irs_group",
    "diagnoses.irs_stage",
    "diagnoses.ishak_fibrosis_score",
    "diagnoses.iss_stage",
    "diagnoses.last_known_disease_status",
    "diagnoses.laterality",
    "diagnoses.margin_distance",
    "diagnoses.margins_involved_site",
    "diagnoses.masaoka_stage",
    "diagnoses.max_tumor_bulk_site",
    "diagnoses.medulloblastoma_molecular_classification",
    "diagnoses.melanoma_known_primary",
    "diagnoses.metastasis_at_diagnosis",
    "diagnoses.metastasis_at_diagnosis_site",
    "diagnoses.method_of_diagnosis",
    "diagnoses.micropapillary_features",
    "diagnoses.mitosis_karyorrhexis_index",
    "diagnoses.mitotic_count",
    "diagnoses.morphology",
    "diagnoses.ovarian_specimen_status",
    "diagnoses.ovarian_surface_involvement",
    "diagnoses.papillary_renal_cell_type",
    "diagnoses.pediatric_kidney_staging",
    "diagnoses.peritoneal_fluid_cytological_status",
    "diagnoses.pregnant_at_diagnosis",
    "diagnoses.primary_diagnosis",
    "diagnoses.primary_disease",
    "diagnoses.primary_gleason_grade",
    "diagnoses.prior_malignancy",
    "diagnoses.prior_treatment",
    "diagnoses.progression_or_recurrence",
    "diagnoses.residual_disease",
    "diagnoses.satellite_nodule_present",
    "diagnoses.secondary_gleason_grade",
    "diagnoses.site_of_resection_or_biopsy",
    "diagnoses.sites_of_involvement",
    "diagnoses.sites_of_involvement_count",
    "diagnoses.submitter_id",
    "diagnoses.supratentorial_localization",
    "diagnoses.synchronous_malignancy",
    "diagnoses.tissue_or_organ_of_origin",
    "diagnoses.tumor_burden",
    "diagnoses.tumor_confined_to_organ_of_origin",
    "diagnoses.tumor_depth",
    "diagnoses.tumor_focality",
    "diagnoses.tumor_grade",
    "diagnoses.tumor_grade_category",
    "diagnoses.tumor_of_origin",
    "diagnoses.tumor_regression_grade",
    "diagnoses.uicc_clinical_m",
    "diagnoses.uicc_clinical_n",
    "diagnoses.uicc_clinical_stage",
    "diagnoses.uicc_clinical_t",
    "diagnoses.uicc_pathologic_m",
    "diagnoses.uicc_pathologic_n",
    "diagnoses.uicc_pathologic_stage",
    "diagnoses.uicc_pathologic_t",
    "diagnoses.uicc_staging_system_edition",
    "diagnoses.ulceration_indicator",
    "diagnoses.weiss_assessment_findings",
    "diagnoses.weiss_assessment_score",
    "diagnoses.who_cns_grade",
    "diagnoses.who_nte_grade",
    "diagnoses.wilms_tumor_histologic_subtype",
    "diagnoses.year_of_diagnosis",
    "diagnoses.treatments.chemo_concurrent_to_radiation",
    "diagnoses.treatments.clinical_trial_indicator",
    "diagnoses.treatments.course_number",
    "diagnoses.treatments.days_to_treatment_end",
    "diagnoses.treatments.days_to_treatment_start",
    "diagnoses.treatments.drug_category",
    "diagnoses.treatments.embolic_agent",
    "diagnoses.treatments.initial_disease_status",
    "diagnoses.treatments.lesions_treated_number",
    "diagnoses.treatments.margin_distance",
    "diagnoses.treatments.margin_status",
    "diagnoses.treatments.margins_involved_site",
    "diagnoses.treatments.number_of_cycles",
    "diagnoses.treatments.number_of_fractions",
    "diagnoses.treatments.prescribed_dose",
    "diagnoses.treatments.prescribed_dose_units",
    "diagnoses.treatments.pretreatment",
    "diagnoses.treatments.protocol_identifier",
    "diagnoses.treatments.radiosensitizing_agent",
    "diagnoses.treatments.reason_treatment_ended",
    "diagnoses.treatments.reason_treatment_not_given",
    "diagnoses.treatments.regimen_or_line_of_therapy",
    "diagnoses.treatments.residual_disease",
    "diagnoses.treatments.route_of_administration",
    "diagnoses.treatments.submitter_id",
    "diagnoses.treatments.therapeutic_agents",
    "diagnoses.treatments.therapeutic_level_achieved",
    "diagnoses.treatments.therapeutic_levels_achieved",
    "diagnoses.treatments.therapeutic_target_level",
    "diagnoses.treatments.timepoint_category",
    "diagnoses.treatments.treatment_anatomic_site",
    "diagnoses.treatments.treatment_anatomic_sites",
    "diagnoses.treatments.treatment_arm",
    "diagnoses.treatments.treatment_dose",
    "diagnoses.treatments.treatment_dose_max",
    "diagnoses.treatments.treatment_dose_units",
    "diagnoses.treatments.treatment_duration",
    "diagnoses.treatments.treatment_effect",
    "diagnoses.treatments.treatment_effect_indicator",
    "diagnoses.treatments.treatment_frequency",
    "diagnoses.treatments.treatment_id",
    "diagnoses.treatments.treatment_intent_type",
    "diagnoses.treatments.treatment_or_therapy",
    "diagnoses.treatments.treatment_outcome",
    "diagnoses.treatments.treatment_outcome_duration",
    "diagnoses.treatments.treatment_type",
    "diagnoses.days_to_progression",
    "diagnoses.days_to_death",
    
    "follow_ups.days_to_progression",
    "follow_ups.days_to_recurrence",
    "follow_ups.progression_or_recurrence",
    "follow_ups.days_to_follow_up",
    "follow_ups.days_to_death"
]


def fetch_project_clinical(project):
    """Fetch and flatten all clinical data (including treatments) for one TCGA project."""
    params = {
        "filters": json.dumps({
            "op": "in",
            "content": {
                "field": "project.project_id",
                "value": [project]
            }
        }),
        "expand": "diagnoses.treatments,follow_ups",  # Add follow_ups here
        "fields": ",".join(FIELDS),
        "format": "JSON",
        "size": "500"
    }

    data = []
    page = 1
    while True:
        params["from"] = (page - 1) * 500
        response = requests.get(GDC_API, params=params)
        response.raise_for_status()
        result = response.json()

        hits = result.get("data", {}).get("hits", [])
        if not hits:
            break

        data.extend(hits)
        page += 1

    # Flatten nested structure
    records = []
    for c in data:
        case_id = c.get("case_id")
        sid = c.get("submitter_id")
        
        diagnoses = c.get("diagnoses", [{}])
        follow_ups = c.get("follow_ups", [{}])
        
        
        for diag in diagnoses:
            treatments = diag.get("treatments", [{}])
            for t in treatments:
                for fu in follow_ups:
                    record = {
                        "cases.case_id": case_id,
                        "cases.submitter_id": sid,
                        #"cases.days_to_lost_to_followup": c.get("days_to_lost_to_followup"),
                        "project.project_id": c.get("project", {}).get("project_id"),
                        **{f"demographic.{k}": v for k, v in c.get("demographic", {}).items()},
                        **{f"diagnoses.{k}": v for k, v in diag.items() if k != "treatments"},
                        **{f"treatments.{k}": v for k, v in t.items()},
                        **{f"follow_ups.{k}": v for k, v in fu.items()}
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    out_path = os.path.join(OUTPUT_DIR, f"{project}_clinical.csv")
    df.to_csv(out_path, index=False)
    return project, len(df)


def main():
    print(f"Downloading clinical data for {len(TCGA_PROJECTS)} TCGA projects with {N_WORKERS} workers…")
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(fetch_project_clinical, proj) for proj in TCGA_PROJECTS]

        for f in tqdm(as_completed(futures), total=len(futures)):
            try:
                project, n = f.result()
                print(f"{project}: {n} records saved.")
            except Exception as e:
                print(f"Error for project: {e}")

    print("\nAll downloads complete!")


if __name__ == "__main__":
    main()