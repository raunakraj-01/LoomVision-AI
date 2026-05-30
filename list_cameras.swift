import AVFoundation

let devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .externalUnknown], mediaType: .video, position: .unspecified).devices
for (index, device) in devices.enumerated() {
    print("[\(index)] \(device.localizedName)")
}
