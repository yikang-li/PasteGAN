# Download all models from the ablation study
mkdir -p sg2im-models

# COCO models
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_gconv.pt -O sg2im-models/coco64_no_gconv.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_relations.pt -O sg2im-models/coco64_no_relations.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_discriminators.pt -O sg2im-models/coco64_no_discriminators.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_img_discriminator.pt -O sg2im-models/coco64_no_img_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_no_obj_discriminator.pt -O sg2im-models/coco64_no_obj_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_gt_layout.pt -O sg2im-models/coco64_gt_layout.pt
wget https://storage.googleapis.com/sg2im-data/small/coco64_gt_layout_no_gconv.pt -O sg2im-models/coco64_gt_layout_no_gconv.pt

# VG models
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_relations.pt -O sg2im-models/vg64_no_relations.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_gconv.pt -O sg2im-models/vg64_no_gconv.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_discriminators.pt -O sg2im-models/vg64_no_discriminators.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_img_discriminator.pt -O sg2im-models/vg64_no_img_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64_no_obj_discriminator.pt -O sg2im-models/vg64_no_obj_discriminator.pt
