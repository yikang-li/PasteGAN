# Download full versions of all models which include training history
mkdir -p sg2im-models/full

# COCO models
wget https://storage.googleapis.com/sg2im-data/full/coco64.pt -O sg2im-models/full/coco64.pt
wget https://storage.googleapis.com/sg2im-data/full/coco64_no_gconv.pt -O sg2im-models/full/coco64_no_gconv.pt
wget https://storage.googleapis.com/sg2im-data/full/coco64_no_relations.pt -O sg2im-models/full/coco64_no_relations.pt
wget https://storage.googleapis.com/sg2im-data/full/coco64_no_discriminators.pt -O sg2im-models/full/coco64_no_discriminators.pt
wget https://storage.googleapis.com/sg2im-data/full/coco64_no_img_discriminator.pt -O sg2im-models/full/coco64_no_img_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/full/coco64_no_obj_discriminator.pt -O sg2im-models/full/coco64_no_obj_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/full/coco64_gt_layout.pt -O sg2im-models/full/coco64_gt_layout.pt
wget https://storage.googleapis.com/sg2im-data/full/coco64_gt_layout_no_gconv.pt -O sg2im-models/full/coco64_gt_layout_no_gconv.pt

# VG models
wget https://storage.googleapis.com/sg2im-data/full/vg64.pt -O sg2im-models/full/vg64.pt
wget https://storage.googleapis.com/sg2im-data/full/vg128.pt -O sg2im-models/full/vg128.pt
wget https://storage.googleapis.com/sg2im-data/full/vg64_no_relations.pt -O sg2im-models/full/vg64_no_relations.pt
wget https://storage.googleapis.com/sg2im-data/full/vg64_no_gconv.pt -O sg2im-models/full/vg64_no_gconv.pt
wget https://storage.googleapis.com/sg2im-data/full/vg64_no_discriminators.pt -O sg2im-models/full/vg64_no_discriminators.pt
wget https://storage.googleapis.com/sg2im-data/full/vg64_no_img_discriminator.pt -O sg2im-models/full/vg64_no_img_discriminator.pt
wget https://storage.googleapis.com/sg2im-data/full/vg64_no_obj_discriminator.pt -O sg2im-models/full/vg64_no_obj_discriminator.pt
