use crate::back::{Backend, BufferT};
use crate::ten::{BinOp, MatOp, RedOp, Shape, ShapeOp, UnOp};
use ndarray::{ArcArray, IxDyn, ShapeBuilder, StrideShape};
use std::borrow::Cow;
use std::mem::size_of;
use std::rc::Rc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    Adapter, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    ComputePassDescriptor, ComputePipelineDescriptor, Device, DeviceDescriptor, Features, Instance,
    Label, Limits, Maintain, MapMode, PipelineLayout, PowerPreference, Queue,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

trait InvocationDims {
    fn dims(&self) -> [u32; 3];
}

impl InvocationDims for Shape {
    /// Return the invocation
    fn dims(&self) -> [u32; 3] {
        match self[(-3)..] {
            [z, y, x] => [x as _, y as _, z as _],
            [y, x] => [1, x as _, y as _],
            [x] => [1, 1, x as _],
            [] => [1, 1, 1],
            _ => panic!(),
        }
    }
}

impl BufferT<WgpuBackend> for Rc<wgpu::Buffer> {
    fn upload(&self, e: &mut WgpuBackend, n: ArcArray<f32, IxDyn>) {
        let data = n.as_standard_layout();
        let data = data.as_slice().unwrap();

        let mut data_buffer = e.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::COPY_SRC,
        });

        let mut encoder = e.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&data_buffer, 0, self, 0, data_buffer.size());
        e.queue.submit(Some(encoder.finish()));
    }

    fn download(&self, shape: &Shape, e: &mut WgpuBackend) -> ArcArray<f32, IxDyn> {
        let mut recv_buffer = e.device.create_buffer(&BufferDescriptor {
            label: None,
            size: self.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = e.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(self, 0, &recv_buffer, 0, self.size());
        e.queue.submit(Some(encoder.finish()));

        let bufslice = recv_buffer.slice(..);
        println!("Mapping {} to {:?}", recv_buffer.size(), shape);
        let (s, r) = futures_intrusive::channel::shared::oneshot_channel();
        bufslice.map_async(MapMode::Read, move |v| s.send(v).unwrap());
        e.device.poll(Maintain::Wait);

        match pollster::block_on(r.receive()) {
            Some(Ok(_)) => {
                let data = bufslice.get_mapped_range();
                let res = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                drop(bufslice);
                recv_buffer.unmap();
                return ArcArray::from_shape_vec(&shape[..], res).unwrap();
            }
            _ => {
                panic!()
            }
        }
    }
}

#[derive(Debug)]
pub struct WgpuBackend {
    pub inst: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl WgpuBackend {
    pub async fn new() -> Self {
        let inst = Instance::default();

        let adapter = inst
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    features: Features::empty(),
                    limits: Limits {
                        max_buffer_size: 4 * 1024 * 1024 * 1024,
                        max_storage_buffer_binding_size: 2 * 1024 * 1024 * 1024,
                        ..Limits::downlevel_defaults()
                    },
                },
                None,
            )
            .await
            .unwrap();

        Self {
            inst,
            adapter,
            device,
            queue,
        }
    }

    fn uni_buf(&self, x: u32, y: u32, z: u32) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[x, y, z]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn mk_ubuf_layout(&self, sizes: &[usize]) -> BindGroupLayout {
        let entries = sizes
            .iter()
            .enumerate()
            .map(|(i, s)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(*s as _),
                },
                count: None,
            })
            .collect::<Vec<_>>();

        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Label::from("Outputs"),
                entries: &entries,
            })
    }

    fn mk_ibuf_layout(&self, count: u32) -> BindGroupLayout {
        let entries = (0..count)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            })
            .collect::<Vec<_>>();

        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Label::from("Inputs"),
                entries: &entries,
            })
    }

    fn mk_obuf_layout(&self, count: u32) -> BindGroupLayout {
        let entries = (0..count)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(0),
                },
                count: None,
            })
            .collect::<Vec<_>>();

        self.device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Label::from("Outputs"),
                entries: &entries,
            })
    }

    fn mk_pipeline_layout(
        &self,
        icount: u32,
        ocount: u32,
        unis: &[usize],
    ) -> (PipelineLayout, [BindGroupLayout; 3]) {
        let uni_layout = self.mk_ubuf_layout(unis);
        let in_layout = self.mk_ibuf_layout(icount);
        let out_layout = self.mk_obuf_layout(ocount);

        (
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&uni_layout, &in_layout, &out_layout],
                    push_constant_ranges: &[],
                }),
            [uni_layout, in_layout, out_layout],
        )
    }

    fn mk_bind_groups(
        &self,
        [ul, il, ol]: &[BindGroupLayout; 3],
        [ubs, ibs, obs]: [&[&Buffer]; 3],
    ) -> [BindGroup; 3] {
        let ubd = BindGroupDescriptor {
            label: None,
            layout: &ul,
            entries: &ubs
                .iter()
                .enumerate()
                .map(|(i, v)| BindGroupEntry {
                    binding: i as u32,
                    resource: v.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        };
        let ibd = BindGroupDescriptor {
            label: None,
            layout: &il,
            entries: &ibs
                .iter()
                .enumerate()
                .map(|(i, v)| BindGroupEntry {
                    binding: i as u32,
                    resource: v.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        };
        let obd = BindGroupDescriptor {
            label: None,
            layout: &ol,
            entries: &obs
                .iter()
                .enumerate()
                .map(|(i, v)| BindGroupEntry {
                    binding: i as u32,
                    resource: v.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        };
        let bind_uni = self.device.create_bind_group(&ubd);
        let bind_in = self.device.create_bind_group(&ibd);
        let bind_out = self.device.create_bind_group(&obd);

        [bind_uni, bind_in, bind_out]
    }
}

impl Backend for WgpuBackend {
    type Buffer = Rc<wgpu::Buffer>;

    fn buffer(&mut self, shape: &Shape) -> Self::Buffer {
        Rc::new(self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4 * (shape.prod() as u64),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    fn unop(&mut self, op: &UnOp, x: &Self::Buffer, xsh: &Shape, osh: &Shape) -> Self::Buffer {
        assert_eq!(xsh, osh);
        const UNARY: &str = include_str!("../kernel/unop.wgsl");
        let out = self.buffer(osh);
        let op_fun = match op {
            UnOp::Abs => "abs",
            UnOp::Neg => "neg",
            UnOp::Logn => "log",
            UnOp::Exp => "exp",
            UnOp::Relu => "relu",
            UnOp::Sign => "sign",
            UnOp::Gtz => "gtz",
            UnOp::Tanh => "tanh",
            UnOp::Sqrt => "sqrt",
            UnOp::Reci => "reci",
        };

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Owned(UNARY.replace("__OP", op_fun))),
        });

        let (pipeline_layout, bind_lts) = self.mk_pipeline_layout(1, 1, &[12]);

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        let [d1, d2, d3] = osh.dims();
        let uni_buf = self.uni_buf(d1, d2, d3);
        let [ubg, ibg, obg] = self.mk_bind_groups(&bind_lts, [&[&uni_buf], &[x], &[&out]]);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &ubg, &[]);
            cpass.set_bind_group(1, &ibg, &[]);
            cpass.set_bind_group(2, &obg, &[]);
            cpass.insert_debug_marker("UnOp");
            cpass.dispatch_workgroups(d1, d2, d3);
        }
        let commands = encoder.finish();
        let sub = self.queue.submit(Some(commands));
        out
    }

    fn binop(
        &mut self,
        op: &BinOp,
        x: &Self::Buffer,
        xsh: &Shape,
        y: &Self::Buffer,
        ysh: &Shape,
        osh: &Shape,
    ) -> Self::Buffer {
        assert_eq!(xsh, ysh);
        assert_eq!(xsh, osh);
        const BINARY: &str = include_str!("../kernel/binop.wgsl");
        let out = self.buffer(osh);

        let op_fun = match op {
            BinOp::Add => "add",
            BinOp::Mul => "mul",
        };

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Owned(BINARY.replace("__OP", op_fun))),
        });

        let (pipeline_layout, bind_lts) = self.mk_pipeline_layout(2, 1, &[12]);

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        let [d1, d2, d3] = osh.dims();
        let uni_buf = self.uni_buf(d1, d2, d3);

        let [ubg, ibg, obg] = self.mk_bind_groups(&bind_lts, [&[&uni_buf], &[x, y], &[&out]]);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &ubg, &[]);
            cpass.set_bind_group(1, &ibg, &[]);
            cpass.set_bind_group(2, &obg, &[]);
            cpass.insert_debug_marker("BinOp");
            cpass.dispatch_workgroups(d1, d2, d3);
        }
        let commands = encoder.finish();
        let sub = self.queue.submit(Some(commands));
        out
    }

    fn redop(
        &mut self,
        op: &RedOp,
        dim: usize,
        x: &Self::Buffer,
        xsh: &Shape,
        osh: &Shape,
    ) -> Self::Buffer {
        assert_eq!(osh[dim as isize], 1);
        const REDUCE: &str = include_str!("../kernel/reduce.wgsl");
        let out = self.buffer(osh);
        let op_fun = match op {
            RedOp::Sum => "add",
            RedOp::Max => "max",
        };

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Owned(REDUCE.replace("__OP", op_fun))),
        });

        #[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C)]
        struct Params {
            repeat: u32,
            stride: u32,
            size: u32,
            init: f32,
        }

        let (pipeline_layout, bind_lts) = self.mk_pipeline_layout(1, 1, &[12, size_of::<Params>()]);

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        let [d1, d2, d3] = osh.dims();
        let uni_buf = self.uni_buf(d1, d2, d3);
        let params = Params {
            repeat: 1,
            stride: xsh[dim as isize + 1..].iter().cloned().product::<usize>() as u32,
            size: xsh[dim as isize] as _,
            init: match op {
                RedOp::Sum => 0.0,
                RedOp::Max => f32::NEG_INFINITY,
            },
        };
        let param_buf = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let [ubg, ibg, obg] =
            self.mk_bind_groups(&bind_lts, [&[&uni_buf, &param_buf], &[x], &[&out]]);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &ubg, &[]);
            cpass.set_bind_group(1, &ibg, &[]);
            cpass.set_bind_group(2, &obg, &[]);
            cpass.insert_debug_marker("UnOp");
            cpass.dispatch_workgroups(d1, d2, d3);
        }
        let commands = encoder.finish();
        let sub = self.queue.submit(Some(commands));
        out
    }

    fn shapop(&mut self, op: &ShapeOp, x: &Self::Buffer, xsh: &Shape, osh: &Shape) -> Self::Buffer {
        let kernel = match op {
            ShapeOp::Broadcast { .. } => panic!(),
            ShapeOp::Transpose { .. } => {
                include_str!("../kernel/transpose.wgsl")
            }
            _ => panic!()
        };
        let out = self.buffer(osh);

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Borrowed(kernel)),
        });

        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C, align(16))]
        struct Params {
            m: u32,
            n: u32,

            sm: u32,
            sn: u32,
        }

        let (pipeline_layout, bind_lts) = self.mk_pipeline_layout(1, 1, &[size_of::<Params>()]);

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        let mut params = match op {
            ShapeOp::Broadcast { .. } => panic!(),
            ShapeOp::Transpose { from, to } => Params {
                m: xsh[*from as isize] as u32,
                n: xsh[*to as isize] as u32,

                sm: xsh[(*from as isize + 1)..].iter().cloned().product::<usize>() as u32,
                sn: xsh[(*to as isize + 1)..].iter().cloned().product::<usize>() as u32,
            },
            ShapeOp::Flip { dim} => {
                panic!()
            }
        };

        let param_buf = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let [ubg, ibg, obg] = self.mk_bind_groups(&bind_lts, [&[&param_buf], &[x], &[&out]]);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &ubg, &[]);
            cpass.set_bind_group(1, &ibg, &[]);
            cpass.set_bind_group(2, &obg, &[]);
            cpass.insert_debug_marker("MatOp");
            cpass.dispatch_workgroups(params.n, params.m, 1);
        }
        let commands = encoder.finish();
        let sub = self.queue.submit(Some(commands));
        out
    }

    fn matop(
        &mut self,
        op: &MatOp,
        x: &Self::Buffer,
        xsh: &Shape,
        y: &Self::Buffer,
        ysh: &Shape,
        osh: &Shape,
    ) -> Self::Buffer {
        assert_eq!(xsh[-2], ysh[-1]);
        assert_eq!(ysh[-2], xsh[-1]);

        assert_eq!(osh[-2..], [xsh[-2], ysh[-1]]);
        const MATMUL: &str = include_str!("../kernel/matmul.wgsl");
        let out = self.buffer(osh);

        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(Cow::Owned(MATMUL.to_string())),
        });

        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        #[repr(C, align(16))]
        struct Params {
            m: u32,
            n: u32,
            k: u32,
            alpha: f32,
        }

        let (pipeline_layout, bind_lts) = self.mk_pipeline_layout(2, 1, &[size_of::<Params>()]);

        let pipeline = self
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: "main",
            });

        let mut params = Params {
            m: xsh[-2] as _,
            n: ysh[-1] as _,
            k: ysh[-2] as _,
            alpha: 1.0,
        };

        let param_buf = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let [ubg, ibg, obg] = self.mk_bind_groups(&bind_lts, [&[&param_buf], &[x, y], &[&out]]);

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor { label: None });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &ubg, &[]);
            cpass.set_bind_group(1, &ibg, &[]);
            cpass.set_bind_group(2, &obg, &[]);
            cpass.insert_debug_marker("MatOp");
            cpass.dispatch_workgroups(params.n, params.m, 1);
        }
        let commands = encoder.finish();
        let sub = self.queue.submit(Some(commands));
        out
    }
}
