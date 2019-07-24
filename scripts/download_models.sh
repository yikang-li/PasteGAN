# Download the main models: 64 x 64 for coco and vg, and 128 x 128 for vg
mkdir -p sg2im-models
wget https://storage.googleapis.com/sg2im-data/small/coco64.pt -O sg2im-models/coco64.pt
wget https://storage.googleapis.com/sg2im-data/small/vg64.pt -O sg2im-models/vg64.pt
wget https://storage.googleapis.com/sg2im-data/small/vg128.pt -O sg2im-models/vg128.pt
