from copy import deepcopy
import torch
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DConditionModel, StableDiffusionPipeline
from transformers import CLIPTextModel
from utils.model_util import   GEGLU
from utils.ConcptPrune_neuron_remover import    NeuronRemover
class ModelLoader:
    def __init__(self, model_type="ESD", concept="nudity", device=None):
        self.model_type = model_type
        self.concept = concept
        self.device = device
        self.pipeline = None

    def load_ti2i_pipeline(self):
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32
        )
        if self.model_type == "ESD":
            self._load_esd()
        elif self.model_type == "UCE":
            self._load_uce()
        elif self.model_type == "FMN":
            self._load_fmn()
        elif self.model_type == "SPM":
            self._load_spm()
        elif self.model_type == "RECE":
            self._load_rece()
        elif self.model_type == "AdvUnlearn":
            self._load_advunlearn()
        elif self.model_type == "DoCo":
            self._load_doco()
        elif self.model_type == "MACE":
            self._load_mace()
        elif self.model_type == "ConceptPrune":
            self._load_conceptprune()
        self.pipeline.safety_checker = lambda images, clip_input: (images, [False])
        self.pipeline.to(self.device)
        return self.pipeline

    def load_t2i_pipeline(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float32
        )

        if self.model_type == "ESD":
            self._load_esd()
        elif self.model_type == "UCE":
            self._load_uce()
        elif self.model_type == "FMN":
            self._load_fmn()
        elif self.model_type == "SPM":
            self._load_spm()
        elif self.model_type == "RECE":
            self._load_rece()
        elif self.model_type == "AdvUnlearn":
            self._load_advunlearn()
        elif self.model_type == "DoCo":
            self._load_doco()
        elif self.model_type == "MACE":
            self._load_mace()
        elif self.model_type == "ConceptPrune":
            self._load_conceptprune()

        self.pipeline.safety_checker = lambda images, clip_input: (images, [False])
        self.pipeline.to(self.device)
        return self.pipeline



    def _load_unet_weights(self, path):
        custom_unet = UNet2DConditionModel.from_config(self.pipeline.unet.config)
        custom_unet.load_state_dict(torch.load(path, map_location=self.device))
        custom_unet.to(self.device)
        self.pipeline.unet = custom_unet

    def _load_esd(self):
        paths = {
            "nudity": "REFORGE/mymodel/ESD/ESD-Nudity-Diffusers-UNet-noxattn.pt",
            "object_church": "REFORGE/mymodel/ESD/ESD-Church-Diffusers-UNet-noxattn.pt",
            "object_parachute": "REFORGE/mymodel/ESD/ESD-Parachute-Diffusers-UNet-noxattn.pt",
            "style_vangogh": "REFORGE/mymodel/ESD/ESD-VanGogh-Diffusers-UNet-xattn.pt"
        }
        self._load_unet_weights(paths[self.concept])

    def _load_uce(self):
        paths = {
            "nudity": "REFORGE/mymodel/UCE/UCE-Nudity-Diffusers-UNet.pt",
            "object_church": "REFORGE/mymodel/UCE/Church-sd_1_4.pt",
            "object_parachute": "REFORGE/mymodel/UCE/Parachute-sd_1_4.pt",
            "style_vangogh": "REFORGE/mymodel/UCE/UCE-VanGogh-Diffusers-UNet.pt"
        }
        self._load_unet_weights(paths[self.concept])

    def _load_fmn(self):
        paths = {
            "nudity": "REFORGE/mymodel/FMN/FMN-Nudity-Diffusers-UNet.pt",
            "object_church": "REFORGE/mymodel/FMN/FMN-Church-Diffusers-UNet.pt",
            "object_parachute": "REFORGE/mymodel/FMN/FMN-Parachute-Diffusers-UNet.pt"
        }
        self._load_unet_weights(paths[self.concept])

    def _load_spm(self):
        paths = {
            "nudity": "REFORGE/mymodel/SPM/SPM-Nudity-Diffusers-UNet.pt",
            "object_church": "REFORGE/mymodel/SPM/SPM-Church-Diffusers-UNet.pt",
            "object_parachute": "REFORGE/mymodel/SPM/SPM-Parachute-Diffusers-UNet.pt"
        }
        self._load_unet_weights(paths[self.concept])

    def _load_rece(self):
        paths = {
            "nudity": "REFORGE/mymodel/RECE/nudity_ep2.pt"
        }
        self._load_unet_weights(paths[self.concept])

    def _load_advunlearn(self):
        cache_path = ".cache"
        model_name_or_path = "OPTML-Group/AdvUnlearn"
        subfolders = {
            "nudity": "nudity_unlearned",
            "object_church": "church_unlearned",
            "object_parachute": "parachute_unlearned",
            "style_vangogh": "vangogh_unlearned"
        }
        custom_text_encoder = CLIPTextModel.from_pretrained(
            model_name_or_path,
            subfolder=subfolders[self.concept],
            cache_dir=cache_path
        )
        self.pipeline.text_encoder = custom_text_encoder

    def _load_doco(self):
        paths = {
            "nudity": "REFORGE/mymodel/DoCo/Nudity.bin",
            "object_parachute": "REFORGE/mymodel/DoCo/Parachute.bin",
            "style_vangogh": "REFORGE/mymodel/DoCo/Vangogh.bin"
        }
        ckpt_path = paths[self.concept]


        st = torch.load(ckpt_path, map_location=self.device)


        unet_sd = self.pipeline.unet
        for name, params in unet_sd.named_parameters():
            if name in st['unet']:
                params.data.copy_(st['unet'][name])


        self.pipeline.unet = deepcopy(unet_sd)

    def _load_mace(self):
        ckpt_BASE = "REFORGE/mymodel/MACE"
        target_ckpt = f"{ckpt_BASE}/{self.concept}"
        pipe = StableDiffusionPipeline.from_pretrained(target_ckpt, torch_dtype=torch.float32).to(self.device)
        self.pipeline = pipe

    def _load_conceptprune(self):
        paths = {
            "nudity": "REFORGE/mymodel/ConceptPrune/nudity/Nudity_skilled_neurons_0.01.pt",
            "object_parachute": "REFORGE/mymodel/ConceptPrune/object_parachute/Parachute_skilled_neurons_0.01.pt",
            "style_vangogh": "REFORGE/mymodel/ConceptPrune/style_vangogh/VanGogh_skilled_neurons_0.01.pt"
        }
        target_ckpt = paths[self.concept]

        neuron_remover = NeuronRemover(path_expert_indx=target_ckpt, T=50, n_layers=16, replace_fn=GEGLU,
                                       hook_module='unet')

        pipe = neuron_remover.observe_activation(deepcopy(self.pipeline))

        self.pipeline = pipe