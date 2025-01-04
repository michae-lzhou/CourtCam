from autodistill.detection import CaptionOntology

ontology = CaptionOntology( {
    "Basketball orange ball": "Basketball",
}
)

img_dir_path = "_raw_frames"
lbl_dir_path = "_annotations"

box_threshold = 0.3
text_threshold = 0.3

from autodistill_grounding_dino import GroundingDINO

base_model = GroundingDINO(ontology=ontology, box_threshold=box_threshold,
        text_threshold=text_threshold)
dataset = base_model.label(input_folder=img_dir_path, extension=".jpg",
        output_folder=lbl_dir_path)
