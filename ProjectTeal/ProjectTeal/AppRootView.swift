//
//  AppRootView.swift
//  ProjectTeal
//
//  Created by Soleil Yu on 2025/9/23.
//

import SwiftUI
import UIKit

struct AppRootView: View {
    let environment: AppEnvironment

    init(environment: AppEnvironment = .shared) {
        self.environment = environment
    }

    var body: some View {
        AppRootViewControllerRepresentable(environment: environment)
    }
}

struct AppRootViewControllerRepresentable: UIViewControllerRepresentable {
    let environment: AppEnvironment

    init(environment: AppEnvironment = .shared) {
        self.environment = environment
    }

    func makeUIViewController(context: Context) -> AppRootViewController {
        return AppRootViewController(environment: environment)
    }
    
    func updateUIViewController(_ uiViewController: AppRootViewController, context: Context) {
        // No updates needed for now
    }
}

#Preview {
    AppRootView()
}
