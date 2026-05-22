/*
* Split&Splat - Copyright (c) 2026, MEDIALab, University of Padova.
*
* Author(s):
*  Leonardo Monchieri (leonardo.monchieri@unipd.it)
*  Elena Camuffo (elenacamuffo97@gmail.com)
*  Francesco Barbato (francesco.barbato@dei.unipd.it)
*  Pietro Zanuttigh (zanuttigh@dei.unipd.it)
*  Simone Milani (simone.milani@dei.unipd.it)
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#ifndef PCD_2D_MASK_H_INCLUDED
#define PCD_2D_MASK_H_INCLUDED

#include <torch/extension.h> 
#include <cuda_runtime.h> 

void run_mask_projection_cuda(
    at::Tensor points,   
    at::Tensor depth,
    at::Tensor extr,
    at::Tensor intr,
    float depth_thresh,
    at::Tensor points_2D,
    at::Tensor computed_depth,
    at::Tensor mask
);
 

#endif