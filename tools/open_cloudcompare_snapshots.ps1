$dst = 'D:\CloudCompare\fec_project_open'
New-Item -ItemType Directory -Force -Path $dst | Out-Null

$files = @(
  '\\wsl$\Ubuntu\home\rog\FEC_Project\reports\snapshots\fusion_seq00_1500k_fecunion_clusters.ply',
  '\\wsl$\Ubuntu\home\rog\FEC_Project\reports\snapshots\single_frame_semantic_seq00_000000_fecunion_clusters.ply',
  '\\wsl$\Ubuntu\home\rog\FEC_Project\reports\snapshots\single_frame_csf_seq00_000000_fecunion_clusters.ply'
)

foreach ($file in $files) {
  Copy-Item $file -Destination $dst -Force
}

Start-Process 'D:\CloudCompare\CloudCompare.exe' -ArgumentList @(
  'D:\CloudCompare\fec_project_open\fusion_seq00_1500k_fecunion_clusters.ply',
  'D:\CloudCompare\fec_project_open\single_frame_semantic_seq00_000000_fecunion_clusters.ply',
  'D:\CloudCompare\fec_project_open\single_frame_csf_seq00_000000_fecunion_clusters.ply'
)
