import React, { Suspense, useRef } from "react";
import { Canvas } from "@react-three/fiber";
import {
  OrbitControls,
  useGLTF,
  Environment,
  ContactShadows,
  TransformControls
} from "@react-three/drei";
import * as THREE from "three";

export type DebugChannel = "full" | "albedo" | "roughness" | "metalness" | "normals" | "wireframe";

export interface AssetViewerProps {
  modelUrl: string;
  debugChannel?: DebugChannel;
  showSkeleton?: boolean;
  showSockets?: boolean;
  sockets?: Array<{ name: string; position: [number, number, number] }>;
}

const ModelRenderer: React.FC<AssetViewerProps> = ({
  modelUrl,
  debugChannel = "full",
  showSkeleton = false,
  showSockets = false,
  sockets = []
}) => {
  const { scene } = useGLTF(modelUrl);
  const skeletonRef = useRef<THREE.SkeletonHelper | null>(null);

  scene.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      const mesh = child as THREE.Mesh;
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (mat) {
        if (mat.map) {
          mat.map.colorSpace = THREE.SRGBColorSpace;
          mat.map.needsUpdate = true;
        }
        if (debugChannel === "wireframe") {
          mat.wireframe = true;
        } else {
          mat.wireframe = false;
          if (debugChannel === "albedo" && mat.map) {
            mat.roughness = 1.0;
            mat.metalness = 0.0;
          } else {
            mat.roughness = 0.65;
            mat.metalness = 0.0;
          }
        }
      }
    }
  });

  return (
    <group>
      <primitive object={scene} />
      {showSkeleton && <primitive object={new THREE.SkeletonHelper(scene)} />}

      {showSockets && sockets.map((s, idx) => (
        <group key={idx} position={s.position}>
          <mesh>
            <sphereGeometry args={[0.04, 16, 16]} />
            <meshBasicMaterial color="#00ffcc" wireframe />
          </mesh>
        </group>
      ))}
    </group>
  );
};

export const AssetViewer: React.FC<AssetViewerProps> = (props) => {
  return (
    <div style={{ width: "100%", height: "100%", minHeight: "500px", background: "#11141a", position: "relative" }}>
      <Canvas shadows camera={{ position: [0, 1.5, 3.5], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 10, 5]} intensity={1.5} castShadow shadow-mapSize={2048} />
        <Suspense fallback={null}>
          <ModelRenderer {...props} />
          <Environment preset="city" />
          <ContactShadows position={[0, 0, 0]} opacity={0.65} scale={10} blur={1.5} far={1} />
        </Suspense>
        <OrbitControls makeDefault minDistance={0.5} maxDistance={25} />
      </Canvas>
    </div>
  );
};
