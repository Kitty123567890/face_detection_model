python -u gen_magaface.py --gpu 0 --algo "buffalo_l" --facescrub-root "F:\megaface_testpack_v1.0\facescrub_images" --megaface-root "F:\megaface_testpack_v1.0\megaface_images" --output "./feature_out" --facescrub-lst "F:\megaface_testpack_v1.0\facescrub_lst" --megaface-lst "F:\megaface_testpack_v1.0\megaface_lst" 



python -u remove_noises.py --algo "$ALGO" --feature-dir-input "./feature_out" --feature-dir-out "./feature_out_clean"




# 设置库路径并运行实验
LD_LIBRARY_PATH="/usr/local/lib64:$LD_LIBRARY_PATH" python -u run_experiment.py \
    "$ROOT/feature_out_clean/megaface" \
    "$ROOT/feature_out_clean/facescrub" \
    "_$ALGO.bin" \
    "$ROOT/results/" \
    -s 1000000 \
    -p ../templatelists/facescrub_features_list.json

# 检查实验是否成功
if [ $? -ne 0 ]; then
    echo "错误: MegaFace 评估实验失败"
    cd -
    exit 1
fi

cd -
