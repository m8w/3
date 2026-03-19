// SyphonBridge.mm — Objective-C++ implementation of the Syphon bridge
// Wraps Syphon's Metal server API for use from Swift
//
// SETUP REQUIRED:
//   1. Download Syphon.framework from https://github.com/Syphon/Syphon-Framework/releases
//   2. Add Syphon.framework to your Xcode target's "Frameworks, Libraries, Embedded Content"
//   3. Set "Embed" to "Embed & Sign"
//   4. Add Syphon.framework to Linked Frameworks in Build Phases
//
// The Syphon Metal API:
//   SyphonMetalServer *server = [[SyphonMetalServer alloc] initWithName:@"..." device:device options:nil];
//   [server publishFrameTexture:tex onCommandBuffer:cmdBuf imageRegion:NSMakeRect(0,0,w,h) flipped:NO];

#import "SyphonBridge.h"

// Conditionally import Syphon — comment out if framework not yet installed
// #import <Syphon/Syphon.h>

@interface SyphonBridgeWrapper ()
// @property (nonatomic, strong) SyphonMetalServer *server;
@property (nonatomic, assign) BOOL serverRunning;
@property (nonatomic, strong) id<MTLDevice> metalDevice;
@property (nonatomic, copy)   NSString *serverName;
@property (nonatomic, assign) NSUInteger connectedClients;
@end

@implementation SyphonBridgeWrapper

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device name:(NSString *)name {
    self = [super init];
    if (self) {
        _metalDevice   = device;
        _serverName    = [name copy];
        _serverRunning = NO;
        _connectedClients = 0;
        [self startServer];
    }
    return self;
}

- (void)startServer {
    // Uncomment when Syphon.framework is linked:
    //
    // NSDictionary *options = @{};
    // self.server = [[SyphonMetalServer alloc] initWithName:self.serverName
    //                                                device:self.metalDevice
    //                                               options:options];
    // self.serverRunning = (self.server != nil);
    //
    // NSLog(@"[Syphon] Server started: %@", self.serverName);

    // Placeholder:
    NSLog(@"[Syphon] SyphonBridgeWrapper: link Syphon.framework to activate.");
    _serverRunning = YES;  // Mark as "running" for UI purposes during development
}

- (void)publishTexture:(id<MTLTexture>)texture commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    if (!_serverRunning || texture == nil) return;

    // Uncomment when Syphon.framework is linked:
    //
    // NSRect region = NSMakeRect(0, 0, texture.width, texture.height);
    // [self.server publishFrameTexture:texture
    //                  onCommandBuffer:commandBuffer
    //                      imageRegion:region
    //                          flipped:NO];
    //
    // self.connectedClients = self.server.hasClients ? 1 : 0;  // Approximate
}

- (void)setName:(NSString *)name {
    _serverName = [name copy];
    // [self.server setName:name];
}

- (void)stop {
    // [self.server stop];
    _serverRunning = NO;
    NSLog(@"[Syphon] Server stopped.");
}

- (BOOL)isRunning {
    return _serverRunning;
}

- (NSUInteger)clientCount {
    return _connectedClients;
}

@end
