#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Dehaze Weight Extractor - Works from any directory
修复版去雾权重提取器 - 支持任意目录运行
"""

import torch
import os
import sys
from pathlib import Path

def find_project_root():
    """查找项目根目录"""
    current_dir = Path.cwd()
    
    # 从当前目录向上查找，直到找到包含config.py的目录
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "config.py").exists():
            return parent
    
    # 如果没找到，尝试常见的项目位置
    possible_roots = [
        Path("/root/autodl-tmp/dehaze_detect_project"),
        Path("../"),
        Path("./")
    ]
    
    for root in possible_roots:
        if root.exists() and (root / "config.py").exists():
            return root
    
    return Path.cwd()

def extract_dehaze_weights_fixed():
    """修复版去雾权重提取函数"""
    
    # 找到项目根目录
    project_root = find_project_root()
    print(f"项目根目录: {project_root}")
    
    # 候选权重文件（使用绝对路径）
    candidate_paths = [
        project_root / "artifacts/weights/hfd_lite_hazydet_512_best.pt",
        project_root / "checkpoints/hfd_lite/dehaze_actual_weights.pt",
        project_root / "checkpoints/stage1_v2_final/best.pth", 
        project_root / "checkpoints/stage1_dehaze_naf/best.pth",
        project_root / "backups_before_fix/hfd_lite_hazydet_512_best.pt",
        project_root / "backups_before_fix/hfd_lite_hazydet_512_e50.pt",
        project_root / "backups_before_fix/hfd_lite_hazydet_512_e20.pt",
    ]
    
    # 首先，让我们看看实际存在哪些文件
    print("\n查找实际存在的权重文件...")
    
    all_weights = []
    for pattern in ["**/*.pt", "**/*.pth"]:
        all_weights.extend(project_root.glob(pattern))
    
    print(f"找到所有权重文件 ({len(all_weights)} 个):")
    for weight_file in sorted(all_weights)[:20]:  # 只显示前20个
        size_mb = weight_file.stat().st_size / 1024 / 1024
        print(f"  {weight_file.relative_to(project_root)} ({size_mb:.1f} MB)")
    
    if len(all_weights) > 20:
        print(f"  ... 还有 {len(all_weights) - 20} 个文件")
    
    # 重新构建候选列表，优先选择较大的文件（可能包含完整模型）
    large_weights = [w for w in all_weights if w.stat().st_size > 100 * 1024 * 1024]  # > 100MB
    medium_weights = [w for w in all_weights if 10 * 1024 * 1024 < w.stat().st_size <= 100 * 1024 * 1024]  # 10-100MB
    
    # 按优先级重新排序
    prioritized_candidates = []
    
    # 优先级1: 包含'best'的大文件
    for w in large_weights:
        if 'best' in w.name.lower():
            prioritized_candidates.append(w)
    
    # 优先级2: artifacts目录的文件
    for w in large_weights:
        if 'artifacts' in str(w):
            prioritized_candidates.append(w)
    
    # 优先级3: 其他大文件
    for w in large_weights:
        if w not in prioritized_candidates:
            prioritized_candidates.append(w)
    
    # 优先级4: 中等大小文件
    prioritized_candidates.extend(medium_weights)
    
    out_path = project_root / "checkpoints/hfd_lite/dehaze_actual_weights.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始按优先级尝试提取权重...")
    
    for i, ckpt_path in enumerate(prioritized_candidates[:10]):  # 只尝试前10个
        try:
            size_mb = ckpt_path.stat().st_size / 1024 / 1024
            print(f"\n[{i+1}/10] 尝试: {ckpt_path.name} ({size_mb:.1f} MB)")
            
            # 加载检查点
            ckpt = torch.load(ckpt_path, map_location="cpu")
            
            # 获取状态字典
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    sd = ckpt["model"]
                elif "state_dict" in ckpt:
                    sd = ckpt["state_dict"]
                elif "model_state_dict" in ckpt:
                    sd = ckpt["model_state_dict"]
                else:
                    sd = ckpt
            else:
                sd = ckpt
            
            print(f"  检查点参数数量: {len(sd)}")
            
            # 尝试不同前缀
            possible_prefixes = [
                "dehazing_net.",
                "dehaze_net.", 
                "dehaze.",
                "generator.",
                ""  # 无前缀（可能是纯去雾模型）
            ]
            
            best_match = None
            best_count = 0
            
            for prefix in possible_prefixes:
                if prefix == "":
                    # 检查是否是纯去雾模型
                    conv_layers = {k: v for k, v in sd.items() if 'conv' in k and isinstance(v, torch.Tensor) and len(v.shape) == 4}
                    if len(conv_layers) >= 10:  # 如果有足够多的卷积层
                        match_sd = sd
                        match_count = len(conv_layers)
                    else:
                        continue
                else:
                    match_sd = {k: v for k, v in sd.items() if k.startswith(prefix)}
                    match_count = len(match_sd)
                
                if match_count > best_count:
                    best_match = (prefix, match_sd)
                    best_count = match_count
            
            if not best_match or best_count < 5:
                print(f"  未找到足够的去雾权重 (找到 {best_count} 个)")
                continue
            
            used_prefix, dehaze_sd = best_match
            print(f"  使用前缀: '{used_prefix}' (匹配 {best_count} 个参数)")
            
            # 清理键名
            clean_dehaze_sd = {}
            for key, value in dehaze_sd.items():
                if used_prefix and used_prefix != "":
                    clean_key = key[len(used_prefix):]
                else:
                    clean_key = key
                clean_dehaze_sd[clean_key] = value
            
            # 保存权重
            save_dict = {
                "model": clean_dehaze_sd,
                "source_file": str(ckpt_path),
                "original_prefix": used_prefix,
                "extracted_params": len(clean_dehaze_sd)
            }
            
            torch.save(save_dict, out_path)
            
            print(f"  ✅ 成功提取 {len(clean_dehaze_sd)} 个去雾参数")
            print(f"  保存到: {out_path}")
            
            # 验证保存的文件
            try:
                test_load = torch.load(out_path, map_location="cpu")
                print(f"  ✅ 权重文件验证成功")
                return str(out_path), clean_dehaze_sd
            except Exception as e:
                print(f"  ❌ 权重文件验证失败: {e}")
                continue
                
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            continue
    
    print(f"\n❌ 所有候选文件都无法提取去雾权重")
    return None, None

def main():
    """主函数"""
    print("🔧 修复版去雾权重提取器")
    print("=" * 50)
    
    # 提取权重
    weight_path, weights_dict = extract_dehaze_weights_fixed()
    
    if weight_path:
        print(f"\n🎉 权重提取成功!")
        print(f"📁 权重文件: {weight_path}")
        print(f"📊 参数数量: {len(weights_dict)}")
        
        print(f"\n💡 接下来的步骤:")
        print(f"1. 更新 config.py 中的 PRETRAINED_DEHAZE:")
        print(f"   PRETRAINED_DEHAZE = '{weight_path}'")
        print(f"2. 重新运行可视化或训练")
        
    else:
        print(f"\n❌ 权重提取失败")
        print(f"💡 建议:")
        print(f"1. 检查是否有可用的权重文件")
        print(f"2. 确认权重文件包含去雾网络参数")

if __name__ == "__main__":
    main()