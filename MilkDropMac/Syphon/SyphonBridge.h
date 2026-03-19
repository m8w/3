// SyphonBridge.h — Objective-C wrapper for the Syphon framework
// Syphon shares GPU textures between Mac apps with zero latency
// Framework: https://github.com/Syphon/Syphon-Framework
//
// Integration: Add Syphon.framework to your Xcode project
// (Download from: https://github.com/Syphon/Syphon-Framework/releases)

#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/// Wraps the Syphon Metal server to publish frames to other apps
@interface SyphonBridgeWrapper : NSObject

/// Initialize with a Metal device and a server name visible to Syphon clients
- (nullable instancetype)initWithDevice:(id<MTLDevice>)device name:(NSString *)name;

/// Publish a Metal texture to all Syphon clients
/// Call this after rendering, before present
- (void)publishTexture:(id<MTLTexture>)texture commandBuffer:(id<MTLCommandBuffer>)commandBuffer;

/// Update the server name shown to clients
- (void)setName:(NSString *)name;

/// Stop the Syphon server
- (void)stop;

/// YES if the server is running
@property (nonatomic, readonly) BOOL isRunning;

/// Number of connected clients
@property (nonatomic, readonly) NSUInteger clientCount;

@end

NS_ASSUME_NONNULL_END
