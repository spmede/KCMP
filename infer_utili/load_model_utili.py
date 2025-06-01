
def load_target_model(args):
    if args.target_model == 'llava_v1_5_7b':
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from infer_utili.llava_infer import load_conversation_template

        llava_model_add = '/data/LMmodel/liuhaotian_llava-v1.5-7b' # please change to the actual model location
        llava_model_name = get_model_name_from_path(llava_model_add)
        conv_mode = load_conversation_template(llava_model_name)
        llava_tokenizer, llava_model, llava_image_processor, _ = load_pretrained_model(
            llava_model_add, None, llava_model_name, gpu_id=args.gpu_id
        )
        return {
            "model": llava_model,
            "tokenizer": llava_tokenizer,
            "image_processor": llava_image_processor,
            "conv_mode": conv_mode,
        }

    elif args.target_model == 'MiniGPT4':
        from minigpt4.common.config import Config
        from minigpt4.common.registry import registry
        from infer_utili.minigpt4_infer import CONV_VISION_Vicuna0, CONV_VISION_LLama2

        args.cfg_path = "/data/target_model/VL-MIA/MiniGPT-4/eval_configs/minigpt4_llama2_eval.yaml"  # please change to the actual model location
        args.options = None
        cfg = Config(args)
        model_cfg = cfg.model_cfg
        model_cfg.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_cfg.arch)
        model = model_cls.from_config(model_cfg).to(f'cuda:{args.gpu_id}')
        processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        processor = registry.get_processor_class(processor_cfg.name).from_config(processor_cfg)

        conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2}
        CONV_VISION = conv_dict[model_cfg.model_type]

        return {
            "model": model,
            "processor": processor,
            "CONV_VISION": CONV_VISION,
        }

    elif args.target_model == 'llama_adapter_v2':
        import llama  # local import
        llama_dir = '/data/LMmodel/llama2-7B'  # please change to the actual model location
        adapter_path = '/data/target_model/model_weight/LORA-BIAS-7B-v21.pth'  # please change to the actual model location
        args.adapter_dir = adapter_path

        model, preprocess = llama.load(
            adapter_path,
            llama_dir,
            llama_type="7B",
            device=f'cuda:{args.gpu_id}',
            max_batch_size=32
        )
        model.eval()

        return {
            "model": model,
            "preprocess": preprocess,
        }

    else:
        raise ValueError(f"Unknown target model: {args.target_model}")
