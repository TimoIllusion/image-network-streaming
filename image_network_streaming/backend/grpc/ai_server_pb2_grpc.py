# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import image_network_streaming.backend.grpc.ai_server_pb2 as ai__server__pb2


class AiDetectionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Detect = channel.unary_unary(
            "/aiapp.AiDetectionService/Detect",
            request_serializer=ai__server__pb2.FrameRequest.SerializeToString,
            response_deserializer=ai__server__pb2.DetectionResponse.FromString,
        )


class AiDetectionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Detect(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_AiDetectionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Detect": grpc.unary_unary_rpc_method_handler(
            servicer.Detect,
            request_deserializer=ai__server__pb2.FrameRequest.FromString,
            response_serializer=ai__server__pb2.DetectionResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "aiapp.AiDetectionService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class AiDetectionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Detect(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/aiapp.AiDetectionService/Detect",
            ai__server__pb2.FrameRequest.SerializeToString,
            ai__server__pb2.DetectionResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
